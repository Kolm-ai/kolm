// kolm code - project instruction files (SOTA spec 1.2). Owner CTX.
//
// A repo that ships AGENTS.md or CLAUDE.md has already written down the rules a
// coder is about to break: which package manager to use, which directories are
// generated, what the commit style is. Reading them costs one file read and
// saves a repair round, so the context pack carries them verbatim (head only)
// rather than making every coder rediscover them from the tree.
//
// Priority is list order: the two agent-facing files come first, so when the
// 6 KB budget runs out it is CONTRIBUTING.md that gets cut, not AGENTS.md.

import { toAscii } from './retrieve.js';

/** Total budget for the rendered block. */
export const INSTRUCTIONS_BYTE_CAP = 6 * 1024;

/** Per-file caps, in priority order. */
export const INSTRUCTION_SOURCES = Object.freeze([
  { path: 'AGENTS.md', cap: 3 * 1024 },
  { path: 'CLAUDE.md', cap: 3 * 1024 },
  { path: '.cursorrules', cap: 3 * 1024 },
  { path: '.github/copilot-instructions.md', cap: 3 * 1024 },
  { path: 'CONTRIBUTING.md', cap: 2 * 1024 },
]);

// The header names every file that made it in, so the model can tell a rule it
// must follow from prose the repo happens to contain.
const HEADER_RESERVE = 192;
const MIN_BODY_BYTES = 120;

async function readMaybe(wt, rel) {
  try {
    const text = await wt.readFile(rel);
    return typeof text === 'string' ? text : null;
  } catch {
    // A missing instruction file is the common case, not an error.
    return null;
  }
}

/** Cut `text` to at most `bytes`, on a codepoint boundary, with a marker. */
function capBytes(text, bytes) {
  if (Buffer.byteLength(text, 'utf8') <= bytes) return text;
  const marker = '\n[truncated]';
  // -8 of slack: cutting mid-codepoint can re-encode as a 3-byte U+FFFD.
  const room = Math.max(0, bytes - Buffer.byteLength(marker, 'utf8') - 8);
  return Buffer.from(text, 'utf8').subarray(0, room).toString('utf8') + marker;
}

/**
 * Read the repo's own instruction files, in priority order, ASCII-safe and
 * capped at 6 KB in total.
 *
 * @param {object} wt worktree (readFile only)
 * @param {{maxBytes?:number, sources?:Array<{path:string, cap:number}>}} [opts]
 * @returns {Promise<{files:Array<{path:string, bytes:number}>, text:string}>}
 */
export async function loadInstructions(wt, opts = {}) {
  const empty = { files: [], text: '' };
  if (!wt || typeof wt.readFile !== 'function') return empty;

  const capTotal = Number(opts.maxBytes) > 0 ? Math.trunc(opts.maxBytes) : INSTRUCTIONS_BYTE_CAP;
  const sources = Array.isArray(opts.sources) && opts.sources.length > 0
    ? opts.sources
    : INSTRUCTION_SOURCES;

  let room = capTotal - HEADER_RESERVE;
  const files = [];
  const blocks = [];
  for (const source of sources) {
    if (room < MIN_BODY_BYTES) break;
    const raw = await readMaybe(wt, source.path);
    if (raw === null || raw.trim() === '') continue;
    const perFile = Number(source.cap) > 0 ? Math.trunc(source.cap) : 3 * 1024;
    const separator = `--- ${source.path} ---\n`;
    const bodyRoom = Math.min(perFile, room - Buffer.byteLength(separator, 'utf8') - 1);
    if (bodyRoom < MIN_BODY_BYTES) continue;
    const body = capBytes(toAscii(raw).trim(), bodyRoom);
    const bytes = Buffer.byteLength(body, 'utf8');
    room -= bytes + Buffer.byteLength(separator, 'utf8') + 1;
    files.push({ path: source.path, bytes });
    blocks.push(`${separator}${body}`);
  }

  if (files.length === 0) return empty;
  const header = `PROJECT INSTRUCTIONS (from ${files.map((f) => f.path).join(', ')}; follow them)`;
  const text = capBytes([header, ...blocks].join('\n'), capTotal);
  return { files, text };
}

/**
 * Render a loadInstructions() result (or a plain string) for a model prompt.
 * @param {{text?:string}|string|null} instructions
 * @returns {string}
 */
export function instructionsToPrompt(instructions) {
  if (!instructions) return '';
  if (typeof instructions === 'string') return toAscii(instructions);
  return toAscii(instructions.text || '');
}

export default { loadInstructions, instructionsToPrompt, INSTRUCTIONS_BYTE_CAP };
