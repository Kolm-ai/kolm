// kolm code - lessons carried across runs in one repo (SOTA spec 1.5). Owner CTX.
//
// The second run on a repo should not relearn what the first one paid for: that
// `npm test` is refused here and `node --test` is not, that edit_file cannot
// match this generated file, that l1 always skips because there is no linter.
// Those facts come out of a finished run's own events - no model call, no
// judgement, just recurrence - and go into a small per-repo JSON file that the
// next run renders into the brief.
//
// Everything here is best effort: a missing, unreadable or corrupt lessons file
// means "no lessons", never a failed run. `config.lessons === false` disables
// both halves so an operator can turn the memory off without deleting it.

import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';

import { toAscii } from './context/retrieve.js';

/** Schema version of the on-disk document. */
export const LESSONS_VERSION = 1;
/** How many lessons one repo keeps. Newest wins when the cap bites. */
export const LESSONS_CAP = 12;
/** Budget for the rendered block. */
export const LESSONS_BYTE_CAP = 1536;
/** One lesson is one sentence. */
export const LESSON_TEXT_CAP = 160;
/** At most this many lessons come out of a single run. */
export const LESSON_CANDIDATE_CAP = 5;

const PROMPT_HEADER = 'LESSONS FROM EARLIER RUNS IN THIS REPO';
const DEDUPE_PREFIX = 60;

/** Lessons are on unless the operator turned them off. */
function enabled(config) {
  return config?.lessons !== false;
}

/** Normalised form used for dedupe: case, whitespace and final stop removed. */
function normalize(text) {
  return String(text ?? '')
    .toLowerCase()
    .replace(/\s+/g, ' ')
    .replace(/[.\s]+$/, '')
    .trim();
}

function dedupeKey(text) {
  return normalize(text).slice(0, DEDUPE_PREFIX);
}

function idFor(text) {
  return crypto.createHash('sha256').update(normalize(text)).digest('hex').slice(0, 12);
}

/** One ASCII sentence, never longer than the cap. */
function sentence(text) {
  const clean = toAscii(text).replace(/\s+/g, ' ').trim();
  if (clean.length <= LESSON_TEXT_CAP) return clean;
  return `${clean.slice(0, LESSON_TEXT_CAP - 3).trimEnd()}...`;
}

/** Accept either the on-disk document or a bare array of lessons. */
function asArray(value) {
  if (Array.isArray(value)) return value;
  if (value && Array.isArray(value.lessons)) return value.lessons;
  return [];
}

/**
 * Filename-safe form of a repo key. The raw key is a filesystem path or a URL,
 * so the slug is only for a human reading the directory; an 8-char digest of
 * the raw key is what actually keeps two repos apart.
 * @param {string} repoKey
 * @returns {string}
 */
export function repoKeySlug(repoKey) {
  const raw = String(repoKey ?? 'repo');
  const digest = crypto.createHash('sha256').update(raw).digest('hex').slice(0, 8);
  const slug = raw
    .replace(/[\\/]+/g, '-')
    .replace(/[^A-Za-z0-9._-]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 60);
  return slug ? `${slug}-${digest}` : `repo-${digest}`;
}

/**
 * Where this repo's lessons live.
 * @param {{dataDir?:string}} config
 * @param {string} repoKey as computed by repo-brief.js repoKeyFor()
 * @returns {string} absolute path
 */
export function lessonsPath(config, repoKey) {
  const dataDir = config?.dataDir ? String(config.dataDir) : path.join(process.cwd(), '.kolm');
  return path.join(dataDir, 'code', 'lessons', `${repoKeySlug(repoKey)}.json`);
}

/** Drop anything that is not a well-formed lesson, so a hand-edited file is safe. */
function sanitizeLessons(raw) {
  const out = [];
  for (const entry of asArray(raw)) {
    const text = typeof entry === 'string' ? sentence(entry) : sentence(entry?.text);
    if (!text) continue;
    out.push({
      id: typeof entry?.id === 'string' && entry.id ? entry.id : idFor(text),
      text,
      runId: typeof entry?.runId === 'string' ? entry.runId : null,
      ts: typeof entry?.ts === 'string' ? entry.ts : null,
      hits: Number.isFinite(Number(entry?.hits)) ? Math.max(1, Math.trunc(Number(entry.hits))) : 1,
    });
  }
  return out;
}

/**
 * Read this repo's lessons. Never throws: a missing or corrupt file, or
 * `config.lessons === false`, both read as "no lessons".
 * @param {{dataDir?:string, lessons?:boolean}} config
 * @param {string} repoKey
 * @returns {{version:number, repoKey:string, lessons:Array<{id:string, text:string, runId:string|null, ts:string|null, hits:number}>}}
 */
export function loadLessons(config, repoKey) {
  const key = String(repoKey ?? 'repo');
  const empty = { version: LESSONS_VERSION, repoKey: key, lessons: [] };
  if (!enabled(config)) return empty;
  let parsed;
  try {
    parsed = JSON.parse(fs.readFileSync(lessonsPath(config, key), 'utf8'));
  } catch {
    return empty;
  }
  return {
    version: Number(parsed?.version) || LESSONS_VERSION,
    repoKey: typeof parsed?.repoKey === 'string' ? parsed.repoKey : key,
    lessons: sanitizeLessons(parsed).slice(0, LESSONS_CAP),
  };
}

/**
 * Write this repo's lessons. Temp file + rename so a reader never sees half a
 * document, 0600 so a shared POSIX box cannot read it (Windows ignores the
 * mode; the profile directory ACL is what protects it there). Best effort: an
 * I/O failure here must never fail a finished run.
 * @param {{dataDir?:string, lessons?:boolean}} config
 * @param {string} repoKey
 * @param {Array<object>|{lessons:Array<object>}} lessons
 * @returns {void}
 */
export function saveLessons(config, repoKey, lessons) {
  if (!enabled(config)) return;
  const key = String(repoKey ?? 'repo');
  const file = lessonsPath(config, key);
  const document = {
    version: LESSONS_VERSION,
    repoKey: key,
    lessons: sanitizeLessons(lessons).slice(0, LESSONS_CAP),
  };
  const tmp = `${file}.tmp.${process.pid}.${crypto.randomBytes(4).toString('hex')}`;
  try {
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(tmp, `${JSON.stringify(document, null, 2)}\n`, { encoding: 'utf8', mode: 0o600 });
    try {
      fs.chmodSync(tmp, 0o600);
    } catch {
      // deliberate: chmod is a no-op on Windows, the write already succeeded
    }
    fs.renameSync(tmp, file);
  } catch {
    try {
      fs.rmSync(tmp, { force: true });
    } catch {
      // deliberate: cleanup only
    }
  }
}

/**
 * Fold this run's proposals into what the repo already knew.
 * Dedupe is on the case-insensitive 60-char prefix; a repeat bumps `hits` and
 * takes the newer wording, so a lesson that keeps recurring keeps its place
 * when the 12-lesson cap bites.
 * @param {Array<object>|{lessons:Array<object>}} existing
 * @param {Array<string|object>} proposed
 * @param {{runId?:string, ts?:string, cap?:number}} [opts]
 * @returns {Array<{id:string, text:string, runId:string|null, ts:string|null, hits:number}>}
 */
export function mergeLessons(existing, proposed, opts = {}) {
  const cap = Number(opts.cap) > 0 ? Math.trunc(opts.cap) : LESSONS_CAP;
  const ts = typeof opts.ts === 'string' && opts.ts ? opts.ts : new Date().toISOString();
  const runId = typeof opts.runId === 'string' ? opts.runId : null;

  const byKey = new Map();
  let order = 0;
  for (const lesson of sanitizeLessons(existing)) {
    byKey.set(dedupeKey(lesson.text), { ...lesson, _order: order });
    order += 1;
  }
  for (const lesson of sanitizeLessons(proposed)) {
    const key = dedupeKey(lesson.text);
    const prior = byKey.get(key);
    byKey.set(key, {
      id: idFor(lesson.text),
      text: lesson.text,
      runId: lesson.runId ?? runId,
      ts: lesson.ts ?? ts,
      hits: prior ? prior.hits + 1 : 1,
      _order: order,
    });
    order += 1;
  }

  // Newest first, so a byte cap on the rendered block drops the stalest lesson.
  const merged = [...byKey.values()].sort((a, b) => {
    const at = a.ts || '';
    const bt = b.ts || '';
    if (at !== bt) return bt.localeCompare(at);
    return b._order - a._order;
  });
  return merged.slice(0, cap).map(({ _order, ...lesson }) => lesson);
}

/**
 * Render lessons for a model prompt. ASCII, at most 1.5 KB, '' when empty.
 * @param {Array<object>|{lessons:Array<object>}|null} lessons
 * @returns {string}
 */
export function lessonsToPrompt(lessons) {
  const list = sanitizeLessons(lessons);
  if (list.length === 0) return '';
  const out = [PROMPT_HEADER];
  let bytes = Buffer.byteLength(PROMPT_HEADER, 'utf8');
  for (const lesson of list) {
    const line = `- ${lesson.text}`;
    const cost = Buffer.byteLength(`\n${line}`, 'utf8');
    if (bytes + cost > LESSONS_BYTE_CAP) break;
    bytes += cost;
    out.push(line);
  }
  // A header with nothing under it tells the model nothing; drop the block.
  return out.length > 1 ? out.join('\n') : '';
}

// --------------------------------------------------------------- extraction

/** Store rows carry `payload`; a caller replaying raw events may use `data`. */
function payloadOf(event) {
  if (!event || typeof event !== 'object') return {};
  const payload = event.payload ?? event.data ?? {};
  return payload && typeof payload === 'object' ? payload : {};
}

function detailOf(event) {
  const detail = payloadOf(event).detail;
  return Array.isArray(detail) ? detail : [];
}

/**
 * Propose lessons from one run's events - hard evidence only, no model call.
 *
 * Five sources, in order: a repair reason that recurred across candidates, a
 * `run` tool call the runtime refused or killed, a test command the plan
 * overrode, an edit_file that could not match its old_string, and a stage that
 * skips for a reason belonging to the repo rather than to the patch.
 *
 * @param {Array<{type?:string, payload?:object}>} events
 * @returns {string[]} at most 5 ASCII sentences, each <= 160 chars
 */
export function extractLessonCandidates(events) {
  const rows = Array.isArray(events) ? events : [];
  const out = [];
  const seen = new Set();
  const push = (text) => {
    const line = sentence(text);
    if (!line) return;
    const key = dedupeKey(line);
    if (seen.has(key)) return;
    seen.add(key);
    if (out.length < LESSON_CANDIDATE_CAP) out.push(line);
  };

  // 1. A repair reason two different candidates hit is a property of the repo.
  const repairReasons = new Map();
  for (const event of rows) {
    if (event?.type !== 'repair') continue;
    const payload = payloadOf(event);
    const candidateId = String(payload.candidateId ?? '');
    for (const reason of Array.isArray(payload.reasons) ? payload.reasons : []) {
      const text = sentence(reason);
      if (!text) continue;
      const key = dedupeKey(text);
      if (!repairReasons.has(key)) repairReasons.set(key, { text, candidates: new Set() });
      repairReasons.get(key).candidates.add(candidateId);
    }
  }
  for (const { text, candidates } of repairReasons.values()) {
    if (candidates.size < 2) continue;
    push(`Repairs on ${candidates.size} candidates hit the same problem: ${text}`);
  }

  // 2. A refused or killed `run` is a fact about this repo's allowed commands.
  const runSeen = new Set();
  for (const event of rows) {
    if (event?.type !== 'coder.turn') continue;
    for (const call of detailOf(event)) {
      if (call?.name !== 'run') continue;
      const cmd = sentence(call.input || '');
      const result = String(call.result ?? '');
      if (!cmd || runSeen.has(cmd)) continue;
      if (/^refused:/.test(result)) {
        runSeen.add(cmd);
        push(`The run tool refuses "${cmd}" in this repo; pick an allowed command instead.`);
      } else if (/timed out after/.test(result)) {
        runSeen.add(cmd);
        push(`"${cmd}" timed out under the run tool in this repo; do not rerun it.`);
      }
    }
  }

  // 3. The plan's test command beats whatever detection guessed.
  for (const event of rows) {
    if (event?.type !== 'plan') continue;
    const cmd = sentence(payloadOf(event).test_command || '');
    if (!cmd) continue;
    push(`An earlier plan set the test command to "${cmd}" for this repo.`);
    break;
  }

  // 4. edit_file could not match: the file's exact bytes (CRLF, tabs) differ
  //    from what the model typed, and that is a property of the file.
  const editSeen = new Set();
  for (const event of rows) {
    if (event?.type !== 'coder.turn') continue;
    for (const call of detailOf(event)) {
      if (call?.name !== 'edit_file') continue;
      const miss = /old_string is not present in (\S+)/.exec(String(call.result ?? ''));
      if (!miss) continue;
      const file = miss[1].replace(/[.,;]$/, '');
      if (editSeen.has(file)) continue;
      editSeen.add(file);
      push(`edit_file could not match old_string in ${file}; read it first and copy its exact bytes (it may use CRLF).`);
    }
  }

  // 5. A stage that skips for a repo-level reason will skip again next time.
  const stageSeen = new Set();
  for (const event of rows) {
    if (event?.type !== 'stage.result') continue;
    const payload = payloadOf(event);
    if (payload.decision !== 'skip') continue;
    const stage = String(payload.stage ?? '');
    if (!stage || stageSeen.has(stage)) continue;
    for (const reason of Array.isArray(payload.reasons) ? payload.reasons : []) {
      if (!/no .*command detected|at baseline|baseline/i.test(String(reason))) continue;
      stageSeen.add(stage);
      push(`Stage ${stage} skips in this repo: ${sentence(reason)}`);
      break;
    }
  }

  return out;
}

export default {
  lessonsPath,
  loadLessons,
  saveLessons,
  mergeLessons,
  lessonsToPrompt,
  extractLessonCandidates,
};
