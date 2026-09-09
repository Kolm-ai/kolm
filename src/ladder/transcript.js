// kolm code - transcript compaction for the coder loop (spec 2.1). Owner CODER.
//
// A long coder session dies the same way every time: the transcript grows past
// the model's context window, the provider rejects the call (or silently drops
// the head), and a candidate that was three turns from finishing is lost. The
// cure has to be mechanical - no model call, no summarisation model, nothing
// that can itself fail - so this module rewrites the message list in place with
// two ordered moves:
//
//   1. Stub old tool results. A `read_file` from twenty turns ago is dead
//      weight: the coder either used it or it can read the file again. The stub
//      says which tool produced it and what it named, so re-running it is one
//      obvious step rather than a guess. `edit_file` / `write_file` results are
//      never stubbed - they are short, and together they are the coder's only
//      record of what it has already changed.
//   2. Collapse the oldest half of the conversation into one summary block
//      built from the tool calls themselves (read what, edited what, ran what
//      and with which exit code).
//
// Everything is bounded and deterministic, and the first user message - the
// plan, the repo brief, the retrieval snippets - is never touched, because it
// is the one message the coder cannot reconstruct with a tool.

/** Characters per token for prose and code. Deliberately conservative. */
const CHARS_PER_TOKEN = 3.6;
/** Flat per-block envelope cost (role, type, ids, JSON punctuation). */
const BLOCK_OVERHEAD_TOKENS = 12;
/** How many trailing assistant turns stay untouched by default. */
const DEFAULT_KEEP_RECENT_TURNS = 6;
/** Tool results shorter than this are not worth a stub. */
const DEFAULT_MIN_TOOL_RESULT_CHARS = 400;
/** Tools whose results are the coder's record of its own edits. */
const KEEP_IN_FULL = new Set(['edit_file', 'write_file']);
/** Upper bound on the mechanical summary block. */
const SUMMARY_CAP = 1200;
/** Upper bound on the path/command echoed inside a stub. */
const LABEL_CAP = 120;
/** How many names each bucket of the summary lists before it says "+N more". */
const SUMMARY_BUCKET_CAP = 6;

const CONTINUE_NOTE = 'The turns above were summarised to fit the context window. '
  + 'Re-run any tool whose output you still need, then continue.';

/** JSON length of a value, with a fallback for anything that will not stringify. */
function safeJsonLength(value) {
  if (value === undefined || value === null) return 0;
  try {
    return JSON.stringify(value)?.length ?? 0;
  } catch {
    return String(value).length;
  }
}

/**
 * The text a `tool_result` block carries. The coder always stores a string, but
 * the Anthropic shape also allows an array of blocks, so both are measured.
 */
function toolResultText(block) {
  const content = block?.content;
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content.map((b) => (typeof b === 'string' ? b : String(b?.text ?? ''))).join('');
  }
  return content === undefined || content === null ? '' : String(content);
}

/** Characters one content block contributes to the estimate. */
function blockChars(block) {
  if (typeof block === 'string') return block.length;
  if (!block || typeof block !== 'object') return 0;
  // tool_result strings are measured exactly: they are the largest and most
  // variable part of a coder transcript, so guessing there is not good enough.
  if (block.type === 'tool_result') return toolResultText(block).length;
  if (block.type === 'text') return String(block.text ?? '').length;
  if (block.type === 'tool_use') return String(block.name ?? '').length + safeJsonLength(block.input);
  return safeJsonLength(block);
}

/**
 * Estimate how many tokens a message list costs.
 *
 * Text is counted at `chars / 3.6`; `tool_result` payloads are measured by
 * their exact character count; every content block adds a flat 12-token
 * envelope. Monotonic by construction: appending a message never lowers the
 * result.
 *
 * @param {Array<{role:string, content:string|Array<object>}>} messages
 * @returns {number} estimated tokens
 */
export function estimateTokens(messages) {
  let chars = 0;
  let blocks = 0;
  for (const message of Array.isArray(messages) ? messages : []) {
    const content = message?.content;
    if (typeof content === 'string') {
      chars += content.length;
      blocks += 1;
      continue;
    }
    if (!Array.isArray(content)) {
      blocks += 1;
      chars += safeJsonLength(content);
      continue;
    }
    for (const block of content) {
      blocks += 1;
      chars += blockChars(block);
    }
  }
  return Math.ceil(chars / CHARS_PER_TOKEN) + blocks * BLOCK_OVERHEAD_TOKENS;
}

/** The 200-char identifying argument of a tool call (never file content). */
function callLabel(input) {
  const src = input && typeof input === 'object' ? input : {};
  const value = src.path ?? src.cmd ?? src.pattern ?? src.glob ?? src.summary ?? '';
  return String(value).slice(0, LABEL_CAP);
}

/**
 * Index `tool_use_id -> { name, label }` over every assistant turn, so a
 * `tool_result` block can name the tool that produced it.
 */
function indexToolCalls(messages) {
  const byId = new Map();
  for (const message of messages) {
    if (message?.role !== 'assistant' || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (block?.type !== 'tool_use') continue;
      byId.set(String(block.id ?? ''), {
        name: String(block.name ?? 'tool'),
        label: callLabel(block.input),
      });
    }
  }
  return byId;
}

/**
 * First index that is still "recent": everything before it is older than the
 * last `keepRecentTurns` assistant turns and is therefore eligible for stubbing.
 */
function recentCutoff(messages, keepRecentTurns) {
  const keep = Math.max(0, Math.trunc(Number(keepRecentTurns)));
  if (keep === 0) return messages.length;
  const assistantAt = [];
  for (let i = 0; i < messages.length; i += 1) {
    if (messages[i]?.role === 'assistant') assistantAt.push(i);
  }
  if (assistantAt.length <= keep) return 0;
  return assistantAt[assistantAt.length - keep];
}

/** Shallow clone that leaves the caller's array and message objects untouched. */
function cloneForRewrite(messages) {
  return messages.map((m) => (Array.isArray(m?.content) ? { ...m, content: m.content.slice() } : m));
}

function bucketLine(label, names) {
  if (names.length === 0) return null;
  const shown = names.slice(0, SUMMARY_BUCKET_CAP);
  const more = names.length > shown.length ? `, +${names.length - shown.length} more` : '';
  return `${label} ${shown.join(', ')}${more}`;
}

/**
 * Build the one-line record of a range of turns from the tool calls inside it.
 * No model call: every word comes from the transcript itself.
 */
function summarizeRange(range) {
  const byId = indexToolCalls(range);
  // Sets, not arrays: a file read six times is one name in the summary.
  const reads = new Set();
  const edits = new Set();
  const deletes = new Set();
  const runs = [];

  // Exit codes live in the tool_result, not in the call, so index them first.
  const exitById = new Map();
  for (const message of range) {
    if (!Array.isArray(message?.content)) continue;
    for (const block of message.content) {
      if (block?.type !== 'tool_result') continue;
      const match = /exit code: (\d+|\?)/.exec(toolResultText(block));
      if (match) exitById.set(String(block.tool_use_id ?? ''), match[1]);
    }
  }

  for (const message of range) {
    if (message?.role !== 'assistant' || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (block?.type !== 'tool_use') continue;
      const { name, label } = byId.get(String(block.id ?? '')) ?? { name: String(block.name ?? ''), label: '' };
      if (label === '' && name !== 'run') continue;
      if (name === 'read_file' || name === 'list_files' || name === 'search') reads.add(label);
      else if (name === 'write_file' || name === 'edit_file') edits.add(label);
      else if (name === 'delete_file') deletes.add(label);
      else if (name === 'run') {
        const exit = exitById.get(String(block.id ?? ''));
        runs.push(exit === undefined ? label : `${label} (exit ${exit})`);
      }
    }
  }

  const parts = [
    bucketLine('read', [...reads]),
    bucketLine('edited', [...edits]),
    bucketLine('deleted', [...deletes]),
    bucketLine('ran', runs),
  ].filter(Boolean);
  const body = parts.length === 0 ? 'no tool calls' : parts.join('; ');
  return `[earlier turns: ${body}]`.slice(0, SUMMARY_CAP);
}

/**
 * Where to cut the oldest half so the surviving transcript still starts a fresh
 * assistant turn. Cutting mid-turn would orphan a `tool_result` whose
 * `tool_use` is gone, which some providers reject outright.
 * @returns {number} last index of the collapsed range, or -1 when there is none
 */
function collapseBoundary(messages) {
  const target = Math.max(1, Math.floor((messages.length - 1) / 2));
  for (let c = target; c >= 2; c -= 1) {
    if (messages[c + 1]?.role === 'assistant') return c;
  }
  for (let c = target + 1; c < messages.length - 1; c += 1) {
    if (messages[c + 1]?.role === 'assistant') return c;
  }
  return -1;
}

/**
 * Shrink a coder transcript until it fits `opts.budgetTokens`.
 *
 * The first user message is never touched. Old `tool_result` payloads are
 * replaced with a one-line stub naming the tool and its argument; if that is
 * not enough, the oldest half of the turns collapses into one mechanically
 * built summary block. Returns the original array untouched when nothing
 * needed doing (`note === null`).
 *
 * @param {Array<{role:string, content:string|Array<object>}>} messages
 * @param {{budgetTokens?:number, keepRecentTurns?:number, minToolResultChars?:number}} [opts]
 * @returns {{messages:Array<object>, dropped:number, savedTokens:number, note:string|null}}
 */
export function compactTranscript(messages, opts = {}) {
  const input = Array.isArray(messages) ? messages : [];
  const unchanged = { messages: input, dropped: 0, savedTokens: 0, note: null };

  const budgetTokens = Number(opts.budgetTokens);
  if (!Number.isFinite(budgetTokens) || budgetTokens <= 0) return unchanged;
  if (input.length <= 1) return unchanged;

  const before = estimateTokens(input);
  if (before <= budgetTokens) return unchanged;

  const keepRecentTurns = opts.keepRecentTurns === undefined
    ? DEFAULT_KEEP_RECENT_TURNS
    : Number(opts.keepRecentTurns);
  const minToolResultChars = opts.minToolResultChars === undefined
    ? DEFAULT_MIN_TOOL_RESULT_CHARS
    : Number(opts.minToolResultChars);

  let work = cloneForRewrite(input);
  const byId = indexToolCalls(work);

  // --- 1. stub old tool results, oldest first ------------------------------
  const cutoff = recentCutoff(work, keepRecentTurns);
  let stubbed = 0;
  for (let i = 1; i < cutoff && estimateTokens(work) > budgetTokens; i += 1) {
    const message = work[i];
    if (!message || !Array.isArray(message.content)) continue;
    for (let j = 0; j < message.content.length; j += 1) {
      if (estimateTokens(work) <= budgetTokens) break;
      const block = message.content[j];
      if (block?.type !== 'tool_result') continue;
      const call = byId.get(String(block.tool_use_id ?? '')) ?? { name: 'tool', label: '' };
      if (KEEP_IN_FULL.has(call.name)) continue;
      const text = toolResultText(block);
      if (text.length < minToolResultChars) continue;
      const where = call.label === '' ? call.name : `${call.name} ${call.label}`;
      message.content[j] = {
        ...block,
        content: `[trimmed ${where}: ${text.length} chars; re-run the tool if you need it]`,
      };
      stubbed += 1;
    }
  }

  // --- 2. collapse the oldest half -----------------------------------------
  let collapsed = 0;
  if (estimateTokens(work) > budgetTokens && work.length >= 5) {
    const cut = collapseBoundary(work);
    if (cut >= 2) {
      const candidate = [
        work[0],
        { role: 'assistant', content: [{ type: 'text', text: summarizeRange(work.slice(1, cut + 1)) }] },
        { role: 'user', content: CONTINUE_NOTE },
        ...work.slice(cut + 1),
      ];
      // Only take the collapse when it actually pays for the two messages it adds.
      if (estimateTokens(candidate) < estimateTokens(work)) {
        collapsed = cut;
        work = candidate;
      }
    }
  }

  if (stubbed === 0 && collapsed === 0) return unchanged;

  const after = estimateTokens(work);
  const noteParts = [];
  if (stubbed > 0) noteParts.push(`stubbed ${stubbed} tool result(s)`);
  if (collapsed > 0) noteParts.push(`collapsed ${collapsed} earlier message(s)`);
  const note = `compacted transcript: ${noteParts.join(', ')}; saved ${before - after} of ${before} tokens`;

  return { messages: work, dropped: stubbed + collapsed, savedTokens: before - after, note };
}

export default { estimateTokens, compactTranscript };
