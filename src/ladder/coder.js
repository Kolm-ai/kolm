// kolm code - the coder tool loop (spec 6.2/6.3). Owner D.
//
// One candidate, one worktree, one bounded conversation. The loop is the same
// for every provider because the history is always Anthropic-shaped: an
// assistant turn carries `tool_use` blocks, and the next user turn carries one
// `tool_result` block per `tool_use`, keyed by `tool_use_id`. The
// OpenAI-compatible adapter translates that shape internally, so this file
// never branches on which model it is talking to.
//
// Everything here is bounded on purpose: turns, run output bytes, command
// prefixes, forbidden paths, and the run's cost ledger. A tool that refuses is
// NOT an exception - it returns the refusal text to the model as the tool
// result, because "you may not write that path" is information the coder can
// act on, while a thrown error just loses the candidate.

import { LadderError } from './errors.js';
import { modelTimeoutMs } from './config.js';
import { matchGlobs } from './stages/l0-meta.js';
import { rubricToPrompt } from './rubric.js';
import { briefToPrompt } from './context/repo-brief.js';
import { allowedForShell } from './worktree/local-git.js';
import { describeModel } from './providers/registry.js';
import { estimateTokens, compactTranscript } from './transcript.js';

/** Hard ceiling on any single tool result handed back to the model. */
const TOOL_RESULT_CAP = 16 * 1024;
/** `run` keeps this many bytes of head and the same of tail. */
const RUN_HALF_CAP = 4 * 1024;
const SEARCH_HIT_CAP = 40;
const LIST_FILE_CAP = 400;
/** Lines a whole-file `read_file` returns before it asks for a window. */
const READ_LINE_WINDOW = 400;
// The brief now carries retrieval snippets and project instructions (spec 1.3);
// the transcript budget, not this cap, is what protects a small model.
const BRIEF_CAP = 40 * 1024;
/** How much of the diff the self-review turn shows. */
const SELF_REVIEW_DIFF_CAP = 24 * 1024;
const HINT_CAP = 240;
const MAX_HINTS = 6;
/** Fraction of the context window the transcript may occupy before compaction. */
const BUDGET_FRACTION = 0.75;
/**
 * `ollama:*` resolves to a wildcard catalog entry whose 131072 is the catalog's
 * generic value, not a promise from the local server: a default `ollama serve`
 * truncates at a small `num_ctx`. Assume the conservative 32k instead.
 */
const OLLAMA_CTX_FALLBACK = 32768;
const GENERIC_CTX = 131072;

/** Command prefixes a coder may `run` when the config does not override them. */
export const DEFAULT_ALLOWED_COMMANDS = Object.freeze([
  'npm', 'npx', 'node', 'pnpm', 'yarn',
  'pytest', 'python', 'go', 'cargo', 'make',
  'ls', 'cat', 'grep',
  'git diff', 'git status', 'git log',
]);

const SYSTEM = [
  'You are the kolm code coder. You change one repository, in one worktree, to satisfy one plan.',
  'Work through tools only: nothing you write in prose reaches the repository.',
  'Touch only what the task needs. A patch that edits unrelated files fails the review even if it works.',
  'Never leave a TODO, a stub, or a function that returns a canned value in place of the work.',
  'Read before you write; run the tests before you call finish.',
  'When the task adds or changes behaviour, add or update a test that exercises it, in the style of the tests already in the repository.',
  'Check every acceptance criterion in the plan against your diff before you call finish; the reviewers check them one by one.',
  'Call finish exactly once, alone in a reply of its own, after you have seen the test results, with a summary of what you changed and why.',
].join(' ');

const TOOL_GUIDANCE = [
  'TOOL DISCIPLINE',
  '- To change part of a file that already exists, use edit_file: it replaces one exact,'
    + ' unique old_string with new_string and leaves the rest of the file untouched.',
  '- Use write_file only for a NEW file or a deliberate full rewrite. Rewriting a large file'
    + ' to change three lines is how patches acquire damage.',
  '- Never inspect or edit files through run. read_file, search and list_files exist for that,'
    + ' they cost nothing, and they cannot be defeated by quoting rules.',
  '- run is for tests, typecheck and lint. Nothing else.',
  '- Diagnosis is not the task. If you have read the relevant file, make the change; you can'
    + ' always run the tests afterwards and adjust.',
].join('\n');

const TURN_INPUT_CAP = 200;
const TURN_RESULT_CAP = 300;
const TURN_DETAIL_CAP = 4000;

/** Wall-clock budget for one `run` tool call. */
const RUN_TIMEOUT_MS = 120000;

/** How many consecutive turns without a file change trigger a nudge. */
export const NO_PROGRESS_WINDOW = 6;
/** Tools that count as making progress. */
const WRITING_TOOLS = new Set(['write_file', 'edit_file', 'delete_file']);
/** Tools whose result the model has to read before it can know it is done. */
const READING_TOOLS = new Set(['read_file', 'list_files', 'search', 'run']);
/**
 * How many premature finishes a session answers before it obeys one. A model
 * that ignores three explanations will not read a fourth (live run
 * r_20260909_cb3b87e4 appended finish to every reply for eight turns); past
 * this point the self-review and the cascade are the cheaper judges.
 */
export const MAX_FINISH_REJECTIONS = 3;

/** What a session that has already proved it can call tools is told. */
const IDLE_NUDGE = 'You made no tool call. If the change is complete and the tests pass, call finish now, '
  + 'alone in its own reply. Otherwise use the tools to continue the work.';
/**
 * What a session that has never called a tool is told, together with the switch
 * to the text tool protocol (spec 4.3). A model whose native tool API silently
 * answers prose ("here is the command to do that", live run r_20260909_b4d9777c)
 * usually follows a format it can see written out in the system prompt.
 */
const TEXT_PROTOCOL_NUDGE = 'You made no tool call; the tool call you described was never executed. '
  + 'From now on write every tool call exactly in the <<<tool name="...">>> ... <<<end>>> form given in '
  + 'the system prompt, one call per block, then wait for the result.';

/**
 * Whether a tool result text is the tool saying no. Used to keep a `finish`
 * that rode along with a failed edit from ending the session.
 * @param {string} text
 * @returns {boolean}
 */
export function isToolFailure(text) {
  return /^(error:|refused:|not found:|unknown tool:|timed out after)/.test(String(text ?? ''));
}

/**
 * Turn the two-character sequences \\n, \\t, \\r into real control characters.
 * Small models on the text tool protocol double-escape JSON strings, so an
 * `old_string` arrives as `a\\nb` where the file has a newline. Only these
 * three sequences are touched; `\\d` in a regex stays as it is.
 * @param {string} text
 * @returns {string}
 */
export function unescapeControlSequences(text) {
  return String(text ?? '').replace(/\\([ntr])/g, (m, c) => (c === 'n' ? '\n' : c === 't' ? '\t' : '\r'));
}

/**
 * True when a string was clearly meant to be multi-line but every newline is
 * the literal two characters backslash-n: no real newline, at least two
 * escaped ones.
 * @param {string} text
 * @returns {boolean}
 */
export function looksDoubleEscaped(text) {
  const s = String(text ?? '');
  return !s.includes('\n') && (s.match(/\\n/g) || []).length >= 2;
}

/** The tool set (spec 6.3), in Anthropic `input_schema` shape. */
export const CODER_TOOLS = Object.freeze([
  {
    name: 'read_file',
    description:
      'Read one file from the worktree. Returns its text, or a not-found note. '
      + `A whole-file read stops after ${READ_LINE_WINDOW} lines; pass start_line/end_line `
      + '(1-based, inclusive) to read a specific window of a long file.',
    input_schema: {
      type: 'object',
      required: ['path'],
      properties: {
        path: { type: 'string', description: 'repo-relative path' },
        start_line: { type: 'integer', description: 'first line to return, 1-based' },
        end_line: { type: 'integer', description: 'last line to return, inclusive' },
      },
    },
  },
  {
    name: 'write_file',
    description: 'Create or overwrite one file with the exact content given. Write whole files, not fragments.',
    input_schema: {
      type: 'object',
      required: ['path', 'content'],
      properties: {
        path: { type: 'string', description: 'repo-relative path' },
        content: { type: 'string' },
      },
    },
  },
  {
    name: 'edit_file',
    description:
      'Replace one exact occurrence of old_string with new_string in an existing file. '
      + 'old_string must appear exactly once; include enough surrounding context to make it unique. '
      + 'This is the tool to reach for when changing a file that already exists.',
    input_schema: {
      type: 'object',
      required: ['path', 'old_string', 'new_string'],
      properties: {
        path: { type: 'string', description: 'repo-relative path' },
        old_string: { type: 'string', description: 'exact text to replace; must be unique in the file' },
        new_string: { type: 'string', description: 'text to put in its place' },
      },
    },
  },
  {
    name: 'delete_file',
    description: 'Delete one file from the worktree.',
    input_schema: {
      type: 'object',
      required: ['path'],
      properties: { path: { type: 'string' } },
    },
  },
  {
    name: 'list_files',
    description: 'List repo-relative paths, optionally filtered by a glob such as src/**/*.js.',
    input_schema: {
      type: 'object',
      properties: { glob: { type: 'string' } },
    },
  },
  {
    name: 'search',
    description: 'Regex search across the worktree. Returns path:line:text hits.',
    input_schema: {
      type: 'object',
      required: ['pattern'],
      properties: {
        pattern: { type: 'string', description: 'JavaScript regular expression' },
        glob: { type: 'string' },
      },
    },
  },
  {
    name: 'run',
    description: 'Run one allow-listed shell command in the worktree (tests, typecheck, lint). Output is truncated.',
    input_schema: {
      type: 'object',
      required: ['cmd'],
      properties: { cmd: { type: 'string' } },
    },
  },
  {
    name: 'finish',
    description: 'Declare the work complete. Call this exactly once, last.',
    input_schema: {
      type: 'object',
      required: ['summary'],
      properties: { summary: { type: 'string' } },
    },
  },
]);

/** Count non-overlapping occurrences of `needle` in `haystack`. */
function countOccurrences(haystack, needle) {
  if (needle === '') return 0;
  let count = 0;
  let at = haystack.indexOf(needle);
  while (at !== -1) {
    count += 1;
    at = haystack.indexOf(needle, at + needle.length);
  }
  return count;
}

function capBytes(text, cap) {
  const src = String(text ?? '');
  const bytes = Buffer.byteLength(src, 'utf8');
  if (bytes <= cap) return src;
  const head = Buffer.from(src, 'utf8').subarray(0, Math.max(0, cap - 8)).toString('utf8');
  return `${head}\n[truncated ${bytes - Buffer.byteLength(head, 'utf8')} bytes]`;
}

/**
 * Keep the head and the tail of a long output and say what went missing. Test
 * runners put the failure list at the top and the summary at the bottom, so
 * plain truncation throws away exactly the half that answers "did it pass?".
 */
function headAndTail(text, half) {
  const src = String(text ?? '');
  const buf = Buffer.from(src, 'utf8');
  if (buf.length <= half * 2) return src;
  const head = buf.subarray(0, half).toString('utf8');
  const tail = buf.subarray(buf.length - half).toString('utf8');
  const omitted = buf.length - Buffer.byteLength(head, 'utf8') - Buffer.byteLength(tail, 'utf8');
  return `${head}\n[... ${omitted} bytes omitted from the middle; head and tail kept ...]\n${tail}`;
}

/**
 * Split on '\n' but leave any '\r' attached, so joining the pieces back with
 * '\n' reproduces a CRLF file byte for byte.
 */
function splitLines(text) {
  const parts = text.split('\n');
  // A trailing newline yields a final empty piece that is not a line.
  const trailingNewline = parts.length > 1 && parts[parts.length - 1] === '';
  return { parts, lineCount: trailingNewline ? parts.length - 1 : parts.length, trailingNewline };
}

/**
 * The text `read_file` returns: the whole file when it is short and no window
 * was asked for, otherwise a bounded window with a marker saying how to get
 * the rest.
 */
function readWindow(text, startLine, endLine) {
  const hasWindow = Number.isFinite(startLine) || Number.isFinite(endLine);
  const { parts, lineCount, trailingNewline } = splitLines(text);
  if (!hasWindow) {
    if (lineCount <= READ_LINE_WINDOW) return text;
    const head = parts.slice(0, READ_LINE_WINDOW).join('\n');
    return `${head}\n[... ${lineCount - READ_LINE_WINDOW} more lines; pass start_line/end_line]`;
  }
  const from = Math.max(1, Math.trunc(Number.isFinite(startLine) ? startLine : 1));
  const rawTo = Number.isFinite(endLine) ? Math.trunc(endLine) : from + READ_LINE_WINDOW - 1;
  const to = Math.min(lineCount, Math.max(from, rawTo), from + READ_LINE_WINDOW - 1);
  if (from > lineCount) {
    return `[no such lines: the file has ${lineCount} line(s)]`;
  }
  const body = parts.slice(from - 1, to).join('\n');
  const suffix = trailingNewline && to === lineCount ? '\n' : '';
  return `${body}${suffix}\n[lines ${from}-${to} of ${lineCount}; pass start_line/end_line for more]`;
}

/**
 * Is `cmd` allowed? Returns null when allowed, a reason string when not.
 * @param {string} cmd
 * @param {string[]} [allowed] prefixes
 * @returns {string|null}
 */
export function commandRefusal(cmd, allowed) {
  const list = Array.isArray(allowed) && allowed.length > 0 ? allowed : DEFAULT_ALLOWED_COMMANDS;
  const trimmed = String(cmd ?? '').trim();
  if (trimmed === '') return 'refused: empty command';
  // Chaining would let a denied command ride along behind an allowed one.
  if (/[;&|`$(){}<>]|\n/.test(trimmed)) {
    return `refused: command contains shell metacharacters; run one plain command at a time (allowed prefixes: ${list.join(', ')})`;
  }
  for (const prefix of list) {
    const p = String(prefix).trim();
    if (p === '') continue;
    if (trimmed === p || trimmed.startsWith(`${p} `)) return null;
  }
  return `refused: "${trimmed.split(/\s+/)[0]}" is not an allowed command prefix (allowed: ${list.join(', ')})`;
}

function buildSystem({ plan, rubric, shell, allowedCommands }) {
  const parts = [SYSTEM, ''];
  parts.push(TOOL_GUIDANCE);
  parts.push('');
  parts.push('ENVIRONMENT');
  parts.push(`platform: ${shell?.platform ?? process.platform}`);
  parts.push(`run executes through: ${shellLabel(shell)}`);
  if (shell && shell.posix === false) {
    parts.push('There is no POSIX shell here: ls, cat, grep, pipes and globbing are NOT available to run.');
  }
  parts.push('read_file, list_files and search never go through the shell, so they work the same everywhere.');
  if (Array.isArray(allowedCommands) && allowedCommands.length > 0) {
    parts.push(`allowed run prefixes: ${allowedCommands.join(', ')}`);
  }
  parts.push('');
  parts.push('PLAN');
  parts.push(`title: ${plan?.title ?? '(none)'}`);
  parts.push(`summary: ${plan?.summary ?? '(none)'}`);
  if (Array.isArray(plan?.acceptance) && plan.acceptance.length > 0) {
    parts.push('acceptance criteria (every one of these is checked later):');
    for (const a of plan.acceptance) parts.push(`  - ${a}`);
  }
  if (Array.isArray(plan?.files_expected) && plan.files_expected.length > 0) {
    parts.push(`files expected to change: ${plan.files_expected.join(', ')}`);
  }
  if (Array.isArray(plan?.forbidden_paths) && plan.forbidden_paths.length > 0) {
    parts.push(`forbidden paths (writes here are refused): ${plan.forbidden_paths.join(', ')}`);
  }
  if (plan?.test_command) parts.push(`test command: ${plan.test_command}`);
  if (rubric) {
    parts.push('');
    parts.push('YOUR PATCH WILL BE SCORED AGAINST THIS RUBRIC');
    parts.push(rubricToPrompt(rubric));
  }
  return parts.join('\n');
}

/**
 * Render the coach's steering hints (spec 2.4). Bounded on both axes so a
 * chatty coach can never crowd out the plan.
 * @param {string[]} [hints]
 * @returns {string|null} the block, or null when there is nothing to say
 */
function hintsBlock(hints) {
  const list = (Array.isArray(hints) ? hints : [])
    .filter((h) => typeof h === 'string' && h.trim() !== '')
    .slice(0, MAX_HINTS)
    .map((h) => h.trim().replace(/\s+/g, ' ').slice(0, HINT_CAP));
  if (list.length === 0) return null;
  return ['HINTS FROM THE ORCHESTRATOR', ...list.map((h) => `  - ${h}`)].join('\n');
}

function buildOpening({ plan, brief, baseline, commands, feedback, hints }) {
  const parts = [];
  if (feedback) {
    parts.push('REPAIR TURN - your previous attempt was rejected. Fix exactly this, then finish.');
    if (feedback.stage) parts.push(`rejected at: ${feedback.stage}`);
    for (const reason of (Array.isArray(feedback.reasons) ? feedback.reasons : []).slice(0, 12)) {
      parts.push(`  - ${reason}`);
    }
    for (const hit of (Array.isArray(feedback.hits) ? feedback.hits : []).slice(0, 12)) {
      if (!hit || typeof hit !== 'object') continue;
      parts.push(`  metric: ${hit.key} = ${hit.value} -> ${hit.band}`);
    }
    for (const failing of (Array.isArray(feedback.failing) ? feedback.failing : []).slice(0, 20)) {
      parts.push(`  failing: ${typeof failing === 'string' ? failing : JSON.stringify(failing)}`);
    }
    if (feedback.instructions) parts.push(`  instructions: ${feedback.instructions}`);
    parts.push('');
    parts.push('Keep everything that already works. Do not start over.');
    parts.push('');
  }
  parts.push('TASK');
  parts.push(plan?.summary ? String(plan.summary) : String(plan?.title ?? 'implement the plan'));
  const hintText = hintsBlock(hints);
  if (hintText) {
    parts.push('');
    parts.push(hintText);
  }
  parts.push('');
  parts.push(capBytes(briefToPrompt(brief), BRIEF_CAP));
  if (baseline) {
    parts.push('');
    parts.push('BASELINE (the repo BEFORE your change - do not "fix" these unless the task says so)');
    const tests = baseline.tests || {};
    parts.push(`  tests: passed=${tests.passed ?? '?'} failed=${tests.failed ?? '?'} exit=${tests.exitCode ?? '?'}`);
    for (const f of (Array.isArray(tests.failing) ? tests.failing : []).slice(0, 10)) {
      parts.push(`    already failing: ${typeof f === 'string' ? f : JSON.stringify(f)}`);
    }
  }
  const testCmd = plan?.test_command || commands?.test;
  if (testCmd) {
    parts.push('');
    parts.push(`Run \`${testCmd}\` before you finish.`);
  }
  parts.push('');
  parts.push('Start by reading the files you intend to change.');
  return parts.join('\n');
}

/** The 200-char identifying argument of a tool call (never file content). */
function toolInputSummary(call) {
  const input = call?.input && typeof call.input === 'object' ? call.input : {};
  const value = input.path ?? input.cmd ?? input.pattern ?? input.glob ?? input.summary ?? '';
  return String(value).slice(0, TURN_INPUT_CAP);
}

/**
 * Bounded per-tool-call detail for the `coder.turn` event: what was called,
 * which path/command it named, and the head of what came back. Never the
 * content a write_file carried.
 * @param {Array<{name:string, input?:object}>} calls
 * @param {string[]} results tool result strings, index-aligned with `calls`
 * @returns {Array<{name:string, input:string, result:string}>}
 */
export function turnDetail(calls, results) {
  const out = [];
  for (let i = 0; i < calls.length; i += 1) {
    out.push({
      name: String(calls[i]?.name ?? '?'),
      input: toolInputSummary(calls[i]),
      result: String(results[i] ?? '').slice(0, TURN_RESULT_CAP),
    });
  }
  // Keep the whole event comfortably under 4 KB.
  while (out.length > 1 && JSON.stringify(out).length > TURN_DETAIL_CAP) out.pop();
  return out;
}

/** Human-readable name of the shell `run` uses. */
function shellLabel(shell) {
  if (!shell) return 'the platform default shell';
  if (shell.kind === 'bash') return `bash (${shell.path ?? 'bash'})`;
  if (shell.kind === 'cmd') return 'cmd.exe';
  return shell.path ? `sh (${shell.path})` : '/bin/sh';
}

/**
 * The `run` allow-list for this worktree: the config's list (or the default),
 * minus anything the resolved shell cannot run.
 * @param {object} [wt]
 * @param {object} [config]
 * @returns {string[]}
 */
export function allowedCommandsFor(wt, config) {
  const configured = Array.isArray(config?.coderAllowedCommands) && config.coderAllowedCommands.length > 0
    ? config.coderAllowedCommands
    : DEFAULT_ALLOWED_COMMANDS;
  return allowedForShell([...configured], wt?.shell);
}

/** Execute one tool call; always resolves to a string for the model. */
async function executeTool({ call, wt, plan, config, signal }) {
  const input = call.input && typeof call.input === 'object' ? call.input : {};
  // A text-protocol block whose JSON did not parse must never look like a
  // successful call: a broken `finish` would otherwise end the session with
  // an empty summary and the model would never learn what went wrong.
  if (typeof input.__parse_error === 'string') {
    return `error: ${input.__parse_error} for ${call.name || 'an unnamed tool block'}; resend the block with one valid JSON object between the markers`;
  }
  try {
    switch (call.name) {
      case 'read_file': {
        const text = await wt.readFile(String(input.path ?? ''));
        if (text === null || text === undefined) return `not found: ${input.path}`;
        return readWindow(String(text), Number(input.start_line), Number(input.end_line));
      }
      case 'write_file': {
        const target = String(input.path ?? '');
        const forbidden = matchGlobs(target, plan?.forbidden_paths || []);
        if (forbidden) return `refused: ${target} matches the forbidden path ${forbidden}`;
        let content = String(input.content ?? '');
        let escapedNote = '';
        if (looksDoubleEscaped(content)) {
          // The whole file on one line with "\\n" between statements is the
          // model's escaping slip, not a one-line file.
          content = unescapeControlSequences(content);
          escapedNote = '; note: the content had literal \\n sequences and no real newlines, they were converted to newlines';
        }
        await wt.writeFile(target, content);
        return `wrote ${target} (${Buffer.byteLength(content, 'utf8')} bytes)${escapedNote}`;
      }
      case 'edit_file': {
        const target = String(input.path ?? '');
        const forbidden = matchGlobs(target, plan?.forbidden_paths || []);
        if (forbidden) return `refused: ${target} matches the forbidden path ${forbidden}`;
        const oldString = String(input.old_string ?? '');
        const newString = String(input.new_string ?? '');
        if (oldString === '') return `refused: edit_file needs a non-empty old_string for ${target}`;
        const before = await wt.readFile(target);
        if (before === null || before === undefined) return `not found: ${target}`;
        // Models emit LF; a CRLF file must still match, and the edit must keep
        // the file's own line endings (live run r_20260909_f0586198 burned three
        // turns on "0 occurrences" against a CRLF fixture).
        let needle = oldString;
        let replacement = newString;
        let count = countOccurrences(before, needle);
        if (count === 0 && before.includes('\r\n') && !oldString.includes('\r')) {
          needle = oldString.replace(/\n/g, '\r\n');
          replacement = newString.replace(/\r?\n/g, '\r\n');
          count = countOccurrences(before, needle);
        }
        let escapedNote = '';
        if (count === 0 && /\\[ntr]/.test(oldString)) {
          // Double-escaped block: "\\n" where the file has a newline. Try the
          // unescaped needle (and its CRLF form) before giving up, and apply
          // the same conversion to the replacement so both sides agree.
          const rawNeedle = unescapeControlSequences(oldString);
          const rawReplacement = unescapeControlSequences(newString);
          const variants = [[rawNeedle, rawReplacement]];
          if (before.includes('\r\n') && !rawNeedle.includes('\r')) {
            variants.push([rawNeedle.replace(/\n/g, '\r\n'), rawReplacement.replace(/\r?\n/g, '\r\n')]);
          }
          for (const [n, r] of variants) {
            const c = countOccurrences(before, n);
            if (c > 0) {
              needle = n;
              replacement = r;
              count = c;
              escapedNote = '; note: old_string matched only after its literal \\n sequences were read as newlines, new_string was converted the same way';
              break;
            }
          }
        }
        if (count === 0) {
          return `refused: old_string is not present in ${target} (0 occurrences). `
            + 'Read the file again and copy the exact text, including whitespace.';
        }
        if (count > 1) {
          return `refused: old_string appears ${count} times in ${target}; it must appear exactly once. `
            + 'Add surrounding lines to make it unique.';
        }
        if (needle === replacement) return `no change: old_string and new_string are identical in ${target}`;
        const at = before.indexOf(needle);
        const after = before.slice(0, at) + replacement + before.slice(at + needle.length);
        await wt.writeFile(target, after);
        return `edited ${target} (1 replacement, ${after.length - before.length >= 0 ? '+' : ''}`
          + `${after.length - before.length} chars)${escapedNote}`;
      }
      case 'delete_file': {
        const target = String(input.path ?? '');
        const forbidden = matchGlobs(target, plan?.forbidden_paths || []);
        if (forbidden) return `refused: ${target} matches the forbidden path ${forbidden}`;
        await wt.deleteFile(target);
        return `deleted ${target}`;
      }
      case 'list_files': {
        const files = await wt.listFiles({ glob: input.glob, limit: LIST_FILE_CAP + 1 });
        const shown = files.slice(0, LIST_FILE_CAP);
        const more = files.length > shown.length ? `\n[${files.length - shown.length} more not shown]` : '';
        return shown.length === 0 ? '(no files matched)' : `${shown.join('\n')}${more}`;
      }
      case 'search': {
        // Ask for one more than the cap so we can tell "exactly 40" from "more".
        const hits = await wt.search(String(input.pattern ?? ''), {
          glob: input.glob,
          limit: SEARCH_HIT_CAP + 1,
        });
        if (hits.length === 0) return '(no matches)';
        const shown = hits.slice(0, SEARCH_HIT_CAP);
        const body = shown.map((h) => `${h.path}:${h.line}: ${String(h.text).slice(0, 200)}`).join('\n');
        if (hits.length <= SEARCH_HIT_CAP) return body;
        return `${body}\n[${SEARCH_HIT_CAP} hits shown; further hits were cut - narrow the pattern or pass a glob]`;
      }
      case 'run': {
        const cmd = String(input.cmd ?? '');
        const refusal = commandRefusal(cmd, allowedCommandsFor(wt, config));
        if (refusal) return refusal;
        const res = await wt.exec(cmd, { timeoutMs: RUN_TIMEOUT_MS, signal });
        const body = [
          res?.timedOut
            ? `timed out after ${RUN_TIMEOUT_MS} ms and was killed (exit code: ${res?.exitCode ?? '?'}); do not rerun it`
            : `exit code: ${res?.exitCode ?? '?'}`,
          headAndTail(String(res?.stdout ?? ''), RUN_HALF_CAP),
          headAndTail(String(res?.stderr ?? ''), RUN_HALF_CAP),
        ].join('\n');
        return body;
      }
      case 'finish':
        return 'acknowledged';
      default:
        return `unknown tool: ${call.name}`;
    }
  } catch (err) {
    // path-escape and friends are answers, not crashes: hand the message back.
    return `error: ${err?.message || String(err)}`;
  }
}

/**
 * Context window we are willing to trust for `ref`, in tokens.
 * @param {string|null} ref
 * @returns {number}
 */
export function contextWindowFor(ref) {
  if (!ref) return OLLAMA_CTX_FALLBACK;
  let described = null;
  try {
    described = describeModel(ref);
  } catch {
    described = null;
  }
  const ctx = Number(described?.ctx);
  if (!Number.isFinite(ctx) || ctx <= 0) return OLLAMA_CTX_FALLBACK;
  // The catalog has no per-tag entry for local models, so a 131072 on an
  // `ollama:` ref is the wildcard's placeholder rather than a measured window.
  if (String(ref).startsWith('ollama:') && ctx === GENERIC_CTX) return OLLAMA_CTX_FALLBACK;
  return ctx;
}

/**
 * Token budget for the coder transcript: three quarters of the effective
 * context window, less the room the reply itself needs.
 * @param {string|null} ref coder model ref
 * @param {number} maxTokens the reply cap handed to the provider
 * @param {object} [config] `config.contextTokens` lowers the window further
 * @returns {number} tokens
 */
export function transcriptBudget(ref, maxTokens, config = {}) {
  const configured = Number(config?.contextTokens);
  const ctx = Number.isFinite(configured) && configured > 0
    ? Math.min(contextWindowFor(ref), configured)
    : contextWindowFor(ref);
  const reply = Number.isFinite(Number(maxTokens)) && Number(maxTokens) > 0 ? Number(maxTokens) : 0;
  // A floor keeps a tiny KOLM_CODE_CONTEXT_TOKENS from producing a budget that
  // no transcript can ever meet, which would compact on every single turn.
  return Math.max(1000, Math.floor(ctx * BUDGET_FRACTION) - reply);
}

/**
 * The user turn that makes the coder read its own diff before it walks away
 * (spec 2.2). Mechanical: the diff and the plan's own acceptance criteria.
 */
function buildSelfReview({ plan, diff }) {
  const parts = ['SELF-REVIEW before finish. This is your diff:'];
  parts.push(capBytes(String(diff ?? ''), SELF_REVIEW_DIFF_CAP));
  const acceptance = Array.isArray(plan?.acceptance) ? plan.acceptance : [];
  if (acceptance.length > 0) {
    parts.push('Check every acceptance criterion:');
    for (const a of acceptance.slice(0, 20)) parts.push(`  - ${a}`);
  } else {
    parts.push('Check every acceptance criterion in the plan above.');
  }
  parts.push('Check: no unrelated files, no leftover debug output, tests you added actually run.');
  parts.push('If something is wrong, fix it with tools now. If it is correct, call finish again.');
  return parts.join('\n');
}

/**
 * The prose of a reply, whatever shape the adapter returned it in.
 * @param {object} reply
 * @returns {string}
 */
function replyText(reply) {
  if (typeof reply?.text === 'string' && reply.text.trim() !== '') return reply.text.trim();
  const blocks = Array.isArray(reply?.content) ? reply.content : [];
  return blocks
    .filter((b) => b && b.type === 'text' && typeof b.text === 'string')
    .map((b) => b.text)
    .join('\n')
    .trim();
}

/**
 * The user turn for a finish with nothing changed: either the model stopped
 * early (the common case) or the task really needs no edit, and it has to say
 * which.
 */
function buildEmptyDiffReview({ plan }) {
  const parts = ['SELF-REVIEW before finish: the worktree has NO changes relative to the base.'];
  const files = Array.isArray(plan?.files_expected) ? plan.files_expected.slice(0, 10) : [];
  if (files.length > 0) parts.push(`The plan expects changes in: ${files.join(', ')}.`);
  parts.push('If the task needs code changes, make them now with edit_file or write_file, run the tests, then call finish again.');
  parts.push('If no change is needed, call finish again with a summary that says exactly why.');
  return parts.join('\n');
}

function overBudget(store, runId, config) {
  const cap = Number(config?.maxCostUsd);
  if (!Number.isFinite(cap) || cap <= 0) return false;
  if (!store || !runId || typeof store.costSummary !== 'function') return false;
  return Number(store.costSummary(runId).total_usd) >= cap;
}

/**
 * Run the coder loop for one candidate.
 *
 * @param {object} args
 * @param {string} args.runId
 * @param {string} args.candidateId
 * @param {object} args.wt worktree handle
 * @param {object} args.plan
 * @param {object} [args.brief]
 * @param {object} [args.baseline]
 * @param {object} [args.commands]
 * @param {object} [args.rubric]
 * @param {object} args.coder provider
 * @param {object} args.store
 * @param {object} [args.config]
 * @param {AbortSignal} [args.signal]
 * @param {number} [args.maxTurns] defaults to config.coderTurns
 * @param {object} [args.logger]
 * @param {object} [args.feedback] repair feedback (see runRepair)
 * @param {string[]} [args.hints] steering hints from the orchestrator's coach
 * @returns {Promise<{summary:string, turns:number, finished:boolean, reason:string,
 *   diff:string, changedFiles:Array, costUsd:number, model:string|null, toolCalls:number,
 *   selfReviewed:boolean, compactions:number, protocol:('text'|'native')}>}
 */
export async function runCoder({
  runId,
  candidateId,
  wt,
  plan,
  brief,
  baseline,
  commands,
  rubric,
  coder,
  store,
  config = {},
  signal,
  maxTurns,
  logger,
  feedback = null,
  hints = null,
} = {}) {
  if (!wt) throw new LadderError('bad-arg', 'runCoder needs a worktree');
  if (!coder || typeof coder.chat !== 'function') throw new LadderError('bad-arg', 'runCoder needs a coder provider');
  if (coder.armed === false) {
    throw new LadderError('not-armed', `the coder rung ${coder.id ?? '?'} is not armed`, { ref: coder.id ?? null });
  }

  const turnCap = Math.max(1, Math.trunc(Number(maxTurns ?? config.coderTurns ?? 24)));
  const system = buildSystem({
    plan,
    rubric,
    shell: wt.shell,
    allowedCommands: allowedCommandsFor(wt, config),
  });
  const messages = [{ role: 'user', content: buildOpening({ plan, brief, baseline, commands, feedback, hints }) }];

  const maxTokens = Number(config.coderMaxTokens) > 0 ? Number(config.coderMaxTokens) : 8000;
  const budgetTokens = transcriptBudget(coder.id ?? null, maxTokens, config);
  // Self-review is the one behaviour here that spends a turn on every candidate
  // that succeeds, so the coder takes it only on an explicit opt-in; the
  // operator default lives in config.js (KOLM_CODE_SELF_REVIEW=1).
  const selfReviewOn = config.selfReview === true;

  let turns = 0;
  let toolCallCount = 0;
  let costUsd = 0;
  let summary = '';
  let finished = false;
  let reason = 'max-turns';
  let model = coder.id ?? null;
  let idleTurns = 0;
  let selfReviewed = false;
  let selfReviewAttempted = false;
  let finishRejections = 0;
  let compactions = 0;
  // Progress watchdog: a coder that reads and greps forever is a coder that
  // never lands a patch. Count turns whose tool calls changed no file.
  let turnsWithoutChange = 0;
  let nudged = 0;
  // Spec 4.3: a coder whose native tool API answers prose is moved to the text
  // tool protocol for the rest of THIS session. The flag is local on purpose -
  // latching the shared adapter would drag every other candidate of the same
  // rung, and every later run in the process, into the text protocol too.
  let textToolsSession = false;

  for (let turn = 1; turn <= turnCap; turn += 1) {
    if (signal?.aborted) {
      reason = 'aborted';
      break;
    }
    if (overBudget(store, runId, config)) {
      reason = 'budget';
      break;
    }

    turns = turn;

    // Compaction runs before the call, not after a rejection: once the provider
    // has answered 400 the turn is already paid for and the candidate is lost.
    let compacted = false;
    let estTokens = estimateTokens(messages);
    if (estTokens > budgetTokens) {
      const result = compactTranscript(messages, { budgetTokens });
      if (result.note) {
        messages.length = 0;
        messages.push(...result.messages);
        compacted = true;
        compactions += 1;
        estTokens = estimateTokens(messages);
        store?.emit?.(runId, 'coder.compact', {
          candidateId,
          turn,
          dropped: result.dropped,
          savedTokens: result.savedTokens,
          note: result.note,
        });
      }
    }

    const reply = await coder.chat({
      system,
      messages,
      tools: CODER_TOOLS,
      maxTokens,
      temperature: 0.2,
      cacheSystem: true,
      textTools: textToolsSession,
      signal,
      timeoutMs: modelTimeoutMs(config),
    });

    const turnCost = Number(reply?.costUsd) || 0;
    costUsd += turnCost;
    model = coder.id ?? reply?.model ?? model;
    // An adapter already latched to the text protocol (registry entry, operator
    // config, or a 400 it saw earlier) says so on every reply. The session is in
    // text mode whether or not this loop asked for it, so there is nothing to
    // switch and no event to emit.
    if (reply?.meta?.textTools === true) textToolsSession = true;
    const calls = Array.isArray(reply?.toolCalls) ? reply.toolCalls : [];
    if (store && runId) {
      store.addCost({
        runId,
        candidateId,
        stage: 'coder',
        provider: reply?.provider ?? coder.provider ?? null,
        model: reply?.model ?? coder.model ?? null,
        inputTokens: reply?.usage?.input ?? 0,
        outputTokens: reply?.usage?.output ?? 0,
        cacheRead: reply?.usage?.cacheRead ?? 0,
        costUsd: turnCost,
      });
    }
    // Emitted once per turn; when the turn ran tools it carries their bounded
    // detail, which is the difference between a replayable run and a list of
    // tool names.
    const emitTurn = (detail) => {
      if (!store || !runId) return;
      store.emit(runId, 'coder.turn', {
        candidateId,
        turn,
        toolCalls: calls.map((c) => c.name),
        costUsd: turnCost,
        estTokens,
        compacted,
        detail: detail ?? [],
      });
    };

    if (reply?.stopReason === 'refusal') {
      emitTurn(null);
      reason = 'refusal';
      break;
    }

    if (calls.length === 0) {
      // Keep what the model said: a prose reply after a rejected finish is
      // usually "the change is complete", and the operator reading
      // `show --events` should see that rather than an empty turn.
      const prose = replyText(reply).slice(0, TURN_RESULT_CAP);
      emitTurn(prose ? [{ name: 'text', input: '', result: prose }] : null);
      idleTurns += 1;
      if (idleTurns >= 2) {
        reason = 'no-tool-calls';
        break;
      }
      // First idle turn of a session that has never called a tool, while it was
      // offered native tools: the model is describing calls in prose instead of
      // emitting them (live run r_20260909_b4d9777c). Repeating the nudge is
      // what failed there, so change the protocol instead of the volume.
      const switchToText = toolCallCount === 0 && !textToolsSession && reply?.meta?.textTools !== true;
      if (switchToText) {
        textToolsSession = true;
        store?.emit?.(runId, 'coder.protocol', {
          candidateId,
          turn,
          protocol: 'text',
          why: 'no-tool-calls',
        });
      }
      messages.push({ role: 'assistant', content: reply?.content ?? [{ type: 'text', text: reply?.text ?? '' }] });
      messages.push({
        role: 'user',
        content: switchToText ? TEXT_PROTOCOL_NUDGE : IDLE_NUDGE,
      });
      continue;
    }
    idleTurns = 0;
    toolCallCount += calls.length;

    messages.push({ role: 'assistant', content: reply.content });
    const results = [];
    let sawFinish = false;
    const resultTexts = [];
    // A finish that rides along with a read while nothing has changed yet, or
    // with a call that failed, was written before the model saw the results
    // (live run r_20260909_1c67825d: read_file + read_file + finish in one
    // reply, empty diff). It is answered, not obeyed. Once the worktree carries
    // a real diff the finish stands: the self-review turn shows the model its
    // diff next to the run output it did not wait for, which is a better
    // correction than another refusal.
    const withReads = calls.some((c) => c.name === 'finish') && calls.some((c) => READING_TOOLS.has(c.name));
    let sawFailure = false;
    let finishRejected = null;
    for (const call of calls) {
      if (call.name === 'finish') {
        let why = null;
        if (finishRejections < MAX_FINISH_REJECTIONS) {
          if (sawFailure) why = 'with-failures';
          else if (withReads && String((await wt.diff()) ?? '').trim() === '') why = 'with-reads';
        }
        if (why) {
          finishRejections += 1;
          finishRejected = why;
          const text = why === 'with-reads'
            ? 'finish ignored: nothing has changed in the worktree, and finish came in the same reply as read_file, list_files, search or run. Read those results, make the change with edit_file, run the tests, then call finish alone in its own reply.'
            : 'finish ignored: a tool call in the same reply failed (see the results above). Fix that first, run the tests, then call finish alone in its own reply.';
          resultTexts.push(text);
          results.push({ type: 'tool_result', tool_use_id: call.id, content: text });
          continue;
        }
        sawFinish = true;
        summary = String(call.input?.summary ?? '').slice(0, 4000);
      }
      // One cap for every tool, applied once, so no new tool can smuggle a
      // megabyte into the transcript.
      const text = capBytes(await executeTool({ call, wt, plan, config, signal }), TOOL_RESULT_CAP);
      if (call.name !== 'finish' && isToolFailure(text)) sawFailure = true;
      resultTexts.push(text);
      results.push({ type: 'tool_result', tool_use_id: call.id, content: text });
    }
    messages.push({ role: 'user', content: results });
    emitTurn(turnDetail(calls, resultTexts));
    if (finishRejected) {
      store?.emit?.(runId, 'coder.finish_rejected', {
        candidateId, turn, reason: finishRejected, count: finishRejections,
      });
    }

    if (sawFinish) {
      // --- self-review (spec 2.2) ----------------------------------------
      // Exactly once per session: the second finish ends the run whatever the
      // review turned up, so a coder that keeps finding one more thing cannot
      // spin here. An empty diff is reviewed too: a finish with nothing
      // changed is almost always a model that stopped early, and one nudge
      // is cheaper than a cascade round that ends in "empty diff".
      if (selfReviewOn && !selfReviewAttempted) {
        const reviewDiff = String((await wt.diff()) ?? '');
        const emptyDiff = reviewDiff.trim() === '';
        selfReviewAttempted = true;
        const rows = emptyDiff ? [] : await wt.changedFiles();
        const changed = (Array.isArray(rows) ? rows : []).map((r) => String(r?.path ?? r)).slice(0, 50);
        const noRoom = turnCap - turn < 2;
        store?.emit?.(runId, 'coder.self_review', {
          candidateId,
          turn,
          diffBytes: Buffer.byteLength(reviewDiff, 'utf8'),
          changedFiles: changed,
          ...(emptyDiff ? { emptyDiff: true } : {}),
          ...(noRoom ? { skipped: 'no-turns' } : {}),
        });
        if (!noRoom) {
          selfReviewed = true;
          messages.push({
            role: 'user',
            content: emptyDiff ? buildEmptyDiffReview({ plan }) : buildSelfReview({ plan, diff: reviewDiff }),
          });
          continue;
        }
      }
      finished = true;
      reason = 'finish';
      break;
    }

    // --- progress watchdog ---------------------------------------------
    turnsWithoutChange = calls.some((c) => WRITING_TOOLS.has(c.name)) ? 0 : turnsWithoutChange + 1;
    if (turnsWithoutChange >= NO_PROGRESS_WINDOW) {
      if (nudged === 0) {
        nudged = 1;
        turnsWithoutChange = 0;
        store?.emit?.(runId, 'coder.nudge', {
          candidateId,
          turn,
          turnsWithoutChange: NO_PROGRESS_WINDOW,
        });
        messages.push({
          role: 'user',
          content: `${NO_PROGRESS_WINDOW} turns without a file change; make the change now with `
            + 'edit_file or write_file, then run the tests.',
        });
        continue;
      }
      store?.emit?.(runId, 'coder.no-progress', {
        candidateId,
        turn,
        turnsWithoutChange: NO_PROGRESS_WINDOW,
      });
      reason = 'no-progress';
      break;
    }
  }

  if (!finished && reason === 'max-turns') logger?.warn?.(`coder ${candidateId} hit the turn cap (${turnCap})`);
  if (reason === 'no-progress') {
    logger?.warn?.(`coder ${candidateId} stopped: ${NO_PROGRESS_WINDOW * 2} turns without a file change`);
  }

  const diff = await wt.diff();
  const changedFiles = await wt.changedFiles();
  if (!summary) {
    summary = finished
      ? 'coder finished without a summary'
      : `coder stopped: ${reason} (${changedFiles.length} file(s) changed)`;
  }

  if (store && runId && candidateId) {
    store.upsertCandidate({
      runId,
      candidateId,
      diffText: diff,
      summary,
      status: finished ? 'coded' : `stopped-${reason}`,
    });
  }

  return {
    summary,
    turns,
    finished,
    reason,
    diff,
    changedFiles,
    costUsd,
    model,
    toolCalls: toolCallCount,
    selfReviewed,
    compactions,
    protocol: textToolsSession ? 'text' : 'native',
  };
}

/**
 * One bounded repair pass over an existing worktree, with the failure that
 * rejected it injected as the opening message.
 *
 * @param {object} args same as runCoder, plus:
 * @param {{stage?:string, reasons?:string[], hits?:Array, failing?:Array, instructions?:string}} args.feedback
 * @param {string[]} [args.hints] steering hints from the coach (spec 4.3)
 * @param {object} [args.coderOverride] provider that writes this repair instead of
 *   `args.coder`; the orchestrator hands the round to a fallback coder when the
 *   attempt being repaired never managed a tool call (spec 4.3)
 * @returns {Promise<object>} same shape as runCoder
 */
export async function runRepair(args = {}) {
  if (!args.feedback) throw new LadderError('bad-arg', 'runRepair needs feedback');
  const { coderOverride, ...rest } = args;
  return runCoder({
    ...rest,
    ...(coderOverride ? { coder: coderOverride } : {}),
    maxTurns: args.maxTurns ?? args.config?.repairTurns ?? 12,
  });
}

export default {
  runCoder,
  runRepair,
  CODER_TOOLS,
  DEFAULT_ALLOWED_COMMANDS,
  allowedCommandsFor,
  commandRefusal,
  turnDetail,
  contextWindowFor,
  transcriptBudget,
  NO_PROGRESS_WINDOW,
  MAX_FINISH_REJECTIONS,
};
