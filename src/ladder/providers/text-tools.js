// kolm code - text tool protocol (owner PROV, spec 3.1).
//
// Some endpoints have no native tool API at all: Ollama answers 400
// ("does not support tools") for `deepseek-coder-v2`, and many OpenRouter free
// models answer the same for `tools`. Rather than lose those rungs, the adapter
// renders the tool definitions into the system prompt and asks the model to
// write its calls as text, then parses them back into the same Anthropic-shaped
// `tool_use` blocks the rest of the ladder already speaks. The coder's
// transcript, events and receipts are therefore identical whether the call went
// out natively or through this protocol.
//
// The wire format is deliberately not JSON-in-markdown: a bare `<<<tool ...>>>`
// marker survives models that wrap everything in code fences, models that add
// prose after the call, and CRLF transports, all of which broke a naive
// ```json``` convention in practice.
//
// ASCII only: every string here reaches a model prompt and the terminal.

/** Wire-format version. Bump when the markers or the block grammar change. */
export const TEXT_TOOL_PROTOCOL_VERSION = 1;

// A block is `<<<tool name="x">>> {json} <<<end>>>`. Tolerances, in order of how
// often a small model needs them: whitespace inside the markers, single-quoted
// or unquoted names, `<<</end>>>` / `<<<endtool>>>` closers, CRLF, and a code
// fence wrapped around the whole block.
const BLOCK_RE = /<<<[ \t]*tool\b([^>]*?)>>>([\s\S]*?)<<<[ \t]*\/?[ \t]*end(?:_?tool)?[ \t]*>>>/gi;
const OPEN_RE = /<<<[ \t]*tool\b([^>]*?)>>>/i;
const NAME_RE = /name[ \t]*=[ \t]*(?:"([^"]*)"|'([^']*)'|([^\s>]+))/i;
const BARE_NAME_RE = /^[ \t]*(?:"([^"]*)"|'([^']*)'|([A-Za-z0-9_.-]+))[ \t]*$/;
const FENCE_BEFORE_RE = /(?:^|\n)([ \t]*```[A-Za-z0-9_+-]*[ \t]*\r?\n)$/;
const FENCE_AFTER_RE = /^[ \t]*\r?\n?[ \t]*```[ \t]*(?=\r?\n|$)/;

/**
 * First balanced `{...}` run in `text`, parsed.
 * Duplicated from openai-compatible.js on purpose: that module imports this one,
 * so importing back would close a cycle for a 20-line helper.
 * @param {string} text
 * @returns {{ok: boolean, value: object|null}}
 */
function firstJsonObject(text) {
  const start = text.indexOf('{');
  if (start === -1) return { ok: false, value: null };
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let i = start; i < text.length; i += 1) {
    const ch = text[i];
    if (inString) {
      if (escaped) escaped = false;
      else if (ch === '\\') escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) {
        try {
          return { ok: true, value: JSON.parse(text.slice(start, i + 1)) };
        } catch {
          return { ok: false, value: null }; // deliberate: a broken run is a parse error, not a crash
        }
      }
    }
  }
  return { ok: false, value: null };
}

/** Strip one wrapping code fence from a block body. */
function stripFence(body) {
  const trimmed = body.trim();
  const opened = /^```[A-Za-z0-9_+-]*[ \t]*\r?\n/.exec(trimmed);
  if (!opened) return trimmed;
  const rest = trimmed.slice(opened[0].length);
  const closed = /\r?\n[ \t]*```[ \t]*$/.exec(rest);
  return (closed ? rest.slice(0, closed.index) : rest).trim();
}

// The nine characters JSON accepts directly after a backslash inside a string.
// Anything else is an invalid escape and makes the whole object unparseable.
const JSON_ESCAPE_CHARS = '"\\/bfnrtu';
const HEX4_RE = /^[0-9a-fA-F]{4}$/;

/**
 * Re-escape the three slips small models make inside JSON string values.
 *
 * Observed live on `ollama:llama3`: the model writes file content into
 * `old_string` with real U+000A line breaks instead of `\n`, and copies a regex
 * literal such as `/(\d+)([hms])/g` verbatim, leaving a bare `\d` that JSON
 * rejects as an invalid escape. Both are string-interior problems, so the walk
 * only rewrites bytes while it is inside a string literal; structure, numbers
 * and whitespace between tokens are copied untouched.
 *
 * Deliberately narrow: no quote balancing, no comma surgery. A repair that
 * silently changes what a string SAYS is worse for a code-editing agent than a
 * parse error the model can be asked to fix.
 *
 * @param {string} text raw block text
 * @returns {string} text with string-interior control chars and bad escapes fixed
 */
function repairJsonStrings(text) {
  let out = '';
  let inString = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (!inString) {
      // Outside a string nothing is rewritten; a quote opens the next string.
      out += ch;
      if (ch === '"') inString = true;
      continue;
    }
    if (ch === '"') {
      out += ch;
      inString = false;
      continue;
    }
    if (ch === '\\') {
      const next = i + 1 < text.length ? text[i + 1] : '';
      if (next === 'u' && HEX4_RE.test(text.slice(i + 2, i + 6))) {
        out += text.slice(i, i + 6); // a well-formed \uXXXX is left exactly as written
        i += 5;
        continue;
      }
      if (next !== '' && next !== 'u' && JSON_ESCAPE_CHARS.includes(next)) {
        out += ch + next; // already a legal escape: byte-for-byte through
        i += 1;
        continue;
      }
      // Invalid escape (`\d` from a regex, `\u` without four hex digits, or a
      // trailing backslash): double it so the character survives literally.
      out += '\\\\';
      continue;
    }
    if (ch === '\n') {
      out += '\\n';
      continue;
    }
    if (ch === '\r') {
      out += '\\r';
      continue;
    }
    if (ch === '\t') {
      out += '\\t';
      continue;
    }
    const code = ch.charCodeAt(0);
    if (code < 0x20) {
      // Any other C0 control character is illegal raw inside a JSON string.
      out += `\\u00${code.toString(16).padStart(2, '0')}`;
      continue;
    }
    out += ch;
  }
  return out;
}

/**
 * Parse JSON, retrying once with the narrow repairs above when it does not.
 *
 * Strict first: text that already parses is returned exactly as `JSON.parse`
 * built it, so a model that double-escapes correctly (deepseek-coder-v2 writes
 * `\\n`) keeps its bytes and nothing downstream changes. Only on failure is the
 * text repaired - code fence stripped, prose before the first `{` and after the
 * last `}` dropped, then the string-interior walk - and parsed a second time.
 * A second failure reports the ORIGINAL error, because that is the one that
 * describes what the model actually wrote.
 *
 * @param {string} text candidate JSON text
 * @returns {{ok: true, value: *, repaired: boolean}|{ok: false, error: string}}
 */
export function parseLenientJson(text) {
  const source = typeof text === 'string' ? text : '';
  let original = 'arguments are not valid JSON';
  try {
    return { ok: true, value: JSON.parse(source), repaired: false };
  } catch (err) {
    original = err?.message ? String(err.message) : original;
  }
  let candidate = stripFence(source);
  const open = candidate.indexOf('{');
  const close = candidate.lastIndexOf('}');
  // Prose around the object is the other common slip; only trim it when the
  // text has already failed to parse, and only between the outermost braces.
  if (open !== -1 && close > open) candidate = candidate.slice(open, close + 1);
  candidate = repairJsonStrings(candidate);
  try {
    return { ok: true, value: JSON.parse(candidate), repaired: true };
  } catch {
    return { ok: false, error: original }; // deliberate: report the first error, not the repaired one
  }
}

/** Name out of the attribute run between `<<<tool` and `>>>`. */
function nameOf(attrs) {
  const named = NAME_RE.exec(attrs ?? '');
  if (named) return (named[1] ?? named[2] ?? named[3] ?? '').trim();
  const bare = BARE_NAME_RE.exec(attrs ?? '');
  if (bare) return (bare[1] ?? bare[2] ?? bare[3] ?? '').trim();
  return '';
}

/**
 * Parse one block body into tool input.
 * `repaired` is true when the body only parsed after a tolerant path ran, so
 * the caller can count how often the model needed rescuing without turning a
 * successful parse into a reported error.
 * @returns {{input: object, error: string|null, repaired: boolean}}
 */
function parseInput(name, rawBody) {
  const body = stripFence(rawBody);
  if (body === '') return { input: {}, error: null, repaired: false }; // a no-argument call is legal
  let value = null;
  let ok = false;
  let repaired = false;
  // Strict parse plus the three string-interior slips, fences and stray prose.
  const lenient = parseLenientJson(body);
  if (lenient.ok) {
    value = lenient.value;
    ok = true;
    repaired = lenient.repaired;
  }
  if (!ok) {
    // Prose that itself contains braces defeats the first-to-last slice; the
    // balanced scan still finds the leading object.
    const balanced = firstJsonObject(body);
    if (balanced.ok) {
      value = balanced.value;
      ok = true;
      repaired = true;
    }
  }
  if (!ok) {
    // Small models leave a trailing comma before the closing brace more often
    // than they get any other detail wrong.
    try {
      value = JSON.parse(body.replace(/,[ \t\r\n]*([}\]])/g, '$1'));
      ok = true;
      repaired = true;
    } catch {
      ok = false; // deliberate: now it really is a parse error
    }
  }
  if (!ok) {
    return {
      input: { __parse_error: 'arguments are not valid JSON', raw: body },
      error: `tool "${name}": arguments are not valid JSON`,
      repaired: false,
    };
  }
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    return {
      input: { __parse_error: 'arguments must be a JSON object', raw: body },
      error: `tool "${name}": arguments must be a JSON object`,
      repaired: false,
    };
  }
  return { input: value, error: null, repaired };
}

/**
 * Parse a model reply written in the text tool protocol.
 *
 * Everything outside the `<<<tool ...>>> ... <<<end>>>` markers is assistant
 * text. A block whose JSON does not parse still becomes a tool call, with
 * `input: { __parse_error, raw }`, so the coder answers it with an error the
 * model can act on instead of silently dropping the turn.
 *
 * `repaired` counts the blocks whose JSON only parsed after a tolerant path
 * ran. It is a health signal, not an error: a repaired block adds nothing to
 * `errors` and the adapters see the same `toolCalls` they always did. The field
 * is additive - the returned object keeps `text`, `toolCalls` and `errors`
 * exactly as before, so adapters that never read it are unaffected.
 *
 * @param {string} text raw assistant text
 * @returns {{text: string, toolCalls: Array<{id:string, name:string, input:object}>, errors: string[], repaired: number}}
 */
export function parseTextToolCalls(text) {
  const source = typeof text === 'string' ? text : '';
  const toolCalls = [];
  const errors = [];
  const kept = [];
  let repairedCount = 0;
  let cursor = 0;

  const push = (name, rawBody) => {
    const id = `text_${toolCalls.length + 1}`;
    if (name === '') {
      errors.push(`tool block ${id} has no name`);
      toolCalls.push({
        id,
        name: '',
        input: { __parse_error: 'tool block has no name', raw: String(rawBody ?? '').trim() },
      });
      return;
    }
    const { input, error, repaired } = parseInput(name, String(rawBody ?? ''));
    if (error) errors.push(error);
    if (repaired) repairedCount += 1;
    toolCalls.push({ id, name, input });
  };

  BLOCK_RE.lastIndex = 0;
  let match = BLOCK_RE.exec(source);
  while (match !== null) {
    let start = match.index;
    let end = BLOCK_RE.lastIndex;
    const before = source.slice(cursor, start);
    const fenceBefore = FENCE_BEFORE_RE.exec(before);
    if (fenceBefore) start -= fenceBefore[1].length;
    // Only swallow the closing fence when there was an opening one, so an
    // unrelated code block later in the reply survives intact.
    if (fenceBefore) {
      const fenceAfter = FENCE_AFTER_RE.exec(source.slice(end));
      if (fenceAfter) end += fenceAfter[0].length;
    }
    kept.push(source.slice(cursor, start));
    cursor = end;
    push(nameOf(match[1]), match[2]);
    match = BLOCK_RE.exec(source);
  }

  // A reply truncated by max_tokens loses its `<<<end>>>`. Treat the tail as one
  // more block rather than throwing the whole call away.
  let tail = source.slice(cursor);
  const dangling = OPEN_RE.exec(tail);
  if (dangling) {
    const bodyStart = dangling.index + dangling[0].length;
    const body = tail.slice(bodyStart);
    tail = tail.slice(0, dangling.index);
    push(nameOf(dangling[1]), body);
  }
  kept.push(tail);

  const outText = kept
    .join('')
    .replace(/\r\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
  return { text: outText, toolCalls, errors, repaired: repairedCount };
}

/** One tool definition as protocol text. */
function renderTool(tool, index) {
  const schema = tool?.input_schema ?? tool?.inputSchema ?? { type: 'object', properties: {} };
  const description = String(tool?.description ?? '').replace(/\s+/g, ' ').trim();
  return [
    `${index}. ${tool?.name ?? '(unnamed)'}${description ? ` - ${description}` : ''}`,
    `   arguments: ${JSON.stringify(schema)}`,
  ].join('\n');
}

/**
 * System-prompt block that teaches the protocol and lists every tool.
 * The adapter appends this after the caller's own system prompt.
 *
 * @param {Array<{name:string, description?:string, input_schema?:object, inputSchema?:object}>} tools
 * @returns {string} ASCII, safe to concatenate into any system prompt
 */
export function renderToolsForText(tools = []) {
  const list = Array.isArray(tools) ? tools : [];
  const lines = [
    `TOOL CALLING PROTOCOL (text mode, v${TEXT_TOOL_PROTOCOL_VERSION})`,
    '',
    'This endpoint has no tool API, so you call a tool by writing it as text.',
    'Write the call in exactly this form, markers included:',
    '',
    '<<<tool name="TOOL_NAME">>>',
    '{"first_argument": "value"}',
    '<<<end>>>',
    '',
    'Rules:',
    '- Between the two markers put ONE JSON object: the tool arguments.',
    '- Use double quotes for every key and every string value. Inside a string',
    '  write a newline as \\n, a tab as \\t and a backslash as \\\\.',
    '- One tool call per block. Several blocks in one reply run in order, top to',
    '  bottom; when in doubt send one block and wait for its result.',
    '- Do not wrap the block in backticks and do not add a language tag.',
    '- Text outside a block is a note to the reader; it never calls a tool.',
    '- After your blocks you get a TOOL RESULTS section; read it, then continue.',
    '- Use only the tool names listed below, spelled exactly as written.',
    '',
    'TOOLS',
  ];
  if (list.length === 0) lines.push('(none available on this turn)');
  else list.forEach((tool, i) => lines.push(renderTool(tool, i + 1)));
  return lines.join('\n');
}

/**
 * Tool results as the user text a text-mode model reads next.
 * `blocks` are Anthropic-shaped `tool_result` blocks; the adapter may attach a
 * `name` to each so the model sees which tool answered.
 *
 * @param {Array<{type?:string, tool_use_id?:string, content?:*, is_error?:boolean, name?:string}>} blocks
 * @returns {string} '' when there is nothing to render
 */
export function renderToolResultsForText(blocks = []) {
  const list = (Array.isArray(blocks) ? blocks : []).filter((b) => b?.type === 'tool_result');
  if (list.length === 0) return '';
  const lines = ['TOOL RESULTS', ''];
  for (const block of list) {
    const attrs = [`id="${block.tool_use_id ?? ''}"`];
    if (block.name) attrs.push(`tool="${block.name}"`);
    if (block.is_error) attrs.push('status="error"');
    lines.push(`<<<result ${attrs.join(' ')}>>>`);
    lines.push(contentToText(block.content));
    lines.push('<<<end>>>');
    lines.push('');
  }
  lines.push('Continue: write the next <<<tool ...>>> block, or give your answer.');
  return lines.join('\n');
}

/** Anthropic-shaped block content -> plain text. */
function contentToText(content) {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = content
      .filter((b) => b?.type === 'text')
      .map((b) => b.text ?? '')
      .join('');
    return text === '' ? JSON.stringify(content) : text;
  }
  if (content === undefined || content === null) return '';
  return JSON.stringify(content);
}

/** One assistant tool call rendered back into protocol text. */
function renderToolCallForText(name, input) {
  let args = '{}';
  try {
    args = JSON.stringify(input ?? {});
  } catch {
    args = '{}'; // deliberate: a cyclic input is not worth failing the turn over
  }
  return `<<<tool name="${name ?? ''}">>>\n${args}\n<<<end>>>`;
}

/**
 * Rewrite an Anthropic-shaped history so no message carries `tool_use` or
 * `tool_result` blocks: the assistant's calls become protocol text and the
 * results become the user text the model expects to read. The transcript the
 * ladder keeps is untouched; only the wire copy is flattened.
 *
 * @param {Array<{role:string, content:*}>} messages
 * @returns {Array<{role:string, content:string}>}
 */
export function toTextProtocolMessages(messages = []) {
  const list = Array.isArray(messages) ? messages : [];
  const out = [];
  // tool_use_id -> tool name, so a result can name the tool that produced it.
  const names = new Map();
  for (const message of list) {
    const role = message?.role ?? 'user';
    const content = message?.content;
    if (typeof content === 'string') {
      out.push({ role, content });
      continue;
    }
    const blocks = Array.isArray(content) ? content : [];
    const parts = [];
    const results = [];
    for (const block of blocks) {
      if (block?.type === 'text') {
        if (block.text) parts.push(String(block.text));
      } else if (block?.type === 'tool_use') {
        names.set(block.id, block.name);
        parts.push(renderToolCallForText(block.name, block.input));
      } else if (block?.type === 'tool_result') {
        results.push({ ...block, name: block.name ?? names.get(block.tool_use_id) ?? null });
      }
    }
    const rendered = renderToolResultsForText(results);
    if (rendered !== '') parts.unshift(rendered);
    out.push({ role, content: parts.join('\n\n') });
  }
  return out;
}

/**
 * System-prompt block for a `jsonSchema` call in text mode: the endpoint has no
 * structured-output field either, so the schema goes into the prompt and the
 * adapter reuses its balanced-JSON extraction on the reply.
 *
 * @param {object} jsonSchema `{name?, schema}` or a bare schema
 * @returns {string}
 */
export function renderJsonOnlyForText(jsonSchema) {
  const schema = jsonSchema?.schema ?? jsonSchema ?? {};
  let rendered = '{}';
  try {
    rendered = JSON.stringify(schema);
  } catch {
    rendered = '{}'; // deliberate: an unserializable schema still gets the JSON-only rule
  }
  return [
    'OUTPUT FORMAT',
    '',
    'Reply with ONE JSON object and nothing else: no prose before it, no prose',
    'after it, no backticks, no language tag. It must satisfy this JSON schema:',
    rendered,
  ].join('\n');
}

/**
 * Does `ref` appear in a forced text-tools list (KOLM_CODE_TEXT_TOOLS)?
 * Entries are exact refs or a `provider:prefix*` wildcard.
 *
 * @param {string} ref normalized model ref
 * @param {string[]} [list] config.textTools
 * @returns {boolean}
 */
export function refForcedToTextTools(ref, list) {
  if (!Array.isArray(list) || list.length === 0) return false;
  const lowered = String(ref ?? '').trim().toLowerCase();
  if (lowered === '') return false;
  return list.some((raw) => {
    const item = String(raw ?? '').trim().toLowerCase();
    if (item === '') return false;
    if (item.endsWith('*')) return lowered.startsWith(item.slice(0, -1));
    return item === lowered;
  });
}

// The two shapes an endpoint uses to say it has no tool API: Ollama's wording
// first, then the phrasing OpenRouter and several gateways use.
const TOOLS_UNSUPPORTED_PATTERNS = Object.freeze([
  /does not support tools/i,
  /tool(s|_choice)? (is|are) not supported/i,
]);

/**
 * Is this error a 400 that means "this endpoint has no tool API"?
 * Accepts a raw fetch error (`{status, body}`) or a mapped LadderError
 * (`detail.status`, `detail.body`).
 *
 * @param {*} err
 * @returns {boolean}
 */
export function looksLikeToolsUnsupported(err) {
  if (!err || typeof err !== 'object') return false;
  const status = err.status ?? err.detail?.status ?? null;
  if (status !== 400 && status !== 422) return false;
  const body = err.body ?? err.detail?.body ?? err.error ?? null;
  let bodyText = '';
  if (typeof body === 'string') bodyText = body;
  else if (body) {
    try {
      bodyText = JSON.stringify(body);
    } catch {
      bodyText = ''; // deliberate: an unserializable body just contributes nothing
    }
  }
  const haystack = `${String(err.message ?? '')} ${bodyText}`;
  return TOOLS_UNSUPPORTED_PATTERNS.some((re) => re.test(haystack));
}

export default {
  TEXT_TOOL_PROTOCOL_VERSION,
  renderToolsForText,
  renderToolResultsForText,
  renderJsonOnlyForText,
  toTextProtocolMessages,
  parseTextToolCalls,
  parseLenientJson,
  refForcedToTextTools,
  looksLikeToolsUnsupported,
};
