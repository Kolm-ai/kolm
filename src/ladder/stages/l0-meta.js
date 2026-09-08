// kolm code — L0 patch hygiene (spec 5.6). Owner C.
//
// The cheapest rung: it never runs the repo's tooling, it only reads the patch.
// Empty / unparsable / binary / forbidden-path / secret-bearing patches die here
// for free; oversized patches and TODO markers are handed up as unsure so the
// judge sees them with the rest of the evidence.

import { parseUnifiedDiff, diffStats } from '../diff-parse.js';
import { bandFor, worst } from '../bands.js';

export const id = 'l0-meta';
export const level = 0;
export const version = 1;
export const workers = 8;

/** L0 is the floor of the ladder: everything runs. */
export const gate = () => true;

/** Paths a patch must never touch, on top of `plan.forbidden_paths`. */
export const DEFAULT_FORBIDDEN = Object.freeze(['.env*', '**/*.pem', '**/node_modules/**', '.git/**', '**/*.key']);

export const SECRET_PATTERNS = Object.freeze([
  { kind: 'aws-access-key', re: /AKIA[0-9A-Z]{16}/ },
  { kind: 'anthropic-key', re: /sk-ant-[A-Za-z0-9_-]{20,}/ },
  { kind: 'openai-key', re: /sk-(?:proj-)?[A-Za-z0-9_-]{20,}/ },
  { kind: 'github-token', re: /gh[pousr]_[A-Za-z0-9]{36,}/ },
  { kind: 'slack-token', re: /xox[abp]-[A-Za-z0-9-]{10,}/ },
  { kind: 'private-key-block', re: /-----BEGIN [A-Z ]*PRIVATE KEY-----/ },
  { kind: 'inline-credential', re: /(api[_-]?key|secret|token|password)\s*[:=]\s*['"][A-Za-z0-9_\-/+=]{16,}['"]/i },
]);

const TODO_RE = /\b(TODO|FIXME|XXX|placeholder|not implemented|NotImplemented)\b/i;

/** Files whose long random-looking tokens are never credentials. */
const ENTROPY_SKIP = ['*.lock', 'package-lock.json', '*.snap', '*.min.js'];
const ENTROPY_TOKEN_RE = /[A-Za-z0-9_\-+/=]{32,}/g;
const ENTROPY_BITS = 4.5;

const TEXT_EXT_JSON = ['.json'];
const TEXT_EXT_YAML = ['.yml', '.yaml'];

const MAX_ARTIFACT_ITEMS = 40;

/**
 * Minimal glob -> RegExp. Supports `*`, `?`, `**` and `**​/`.
 * Patterns without a `/` match the basename too (gitignore convention), which
 * is what makes `.env*` catch `config/.env.local`.
 * @param {string} glob
 * @returns {RegExp}
 */
export function globToRegExp(glob) {
  const g = String(glob);
  let re = '';
  for (let i = 0; i < g.length; i += 1) {
    const c = g[i];
    if (c === '*') {
      if (g[i + 1] === '*') {
        if (g[i + 2] === '/') {
          re += '(?:.*/)?';
          i += 2;
        } else {
          re += '.*';
          i += 1;
        }
      } else {
        re += '[^/]*';
      }
    } else if (c === '?') {
      re += '[^/]';
    } else if ('.+^${}()|[]\\'.includes(c)) {
      re += `\\${c}`;
    } else {
      re += c;
    }
  }
  return new RegExp(`^${re}$`);
}

/**
 * @param {string} file posix path
 * @param {string[]} globs
 * @returns {string|null} the pattern that matched, or null
 */
export function matchGlobs(file, globs) {
  const path = String(file || '').replace(/\\/g, '/');
  const base = path.split('/').pop();
  for (const glob of globs || []) {
    const re = globToRegExp(glob);
    if (re.test(path)) return glob;
    if (!String(glob).includes('/') && re.test(base)) return glob;
  }
  return null;
}

/**
 * Shannon entropy in bits per character.
 * @param {string} s
 * @returns {number}
 */
export function shannonEntropy(s) {
  const text = String(s);
  if (text.length === 0) return 0;
  const freq = new Map();
  for (const ch of text) freq.set(ch, (freq.get(ch) || 0) + 1);
  let bits = 0;
  for (const n of freq.values()) {
    const p = n / text.length;
    bits -= p * Math.log2(p);
  }
  return bits;
}

function extOf(file) {
  const base = String(file || '').split('/').pop() || '';
  const dot = base.lastIndexOf('.');
  return dot > 0 ? base.slice(dot).toLowerCase() : '';
}

function redact(match) {
  const s = String(match);
  return s.length <= 8 ? `${s.slice(0, 2)}…` : `${s.slice(0, 6)}…(${s.length} chars)`;
}

function addedLines(file) {
  const out = [];
  for (const hunk of file.hunks || []) {
    let lineNo = hunk.newStart;
    for (const line of hunk.lines) {
      if (line.type === '+') {
        out.push({ line: lineNo, text: line.text });
        lineNo += 1;
      } else if (line.type === ' ') {
        lineNo += 1;
      }
    }
  }
  return out;
}

/**
 * Heuristic YAML sanity check. kolm ships no YAML dependency (testing policy:
 * no new npm deps), so this is NOT a parse: it only catches tab indentation
 * (always invalid in YAML) and an odd number of double quotes on a line.
 * Tabs are reported as an error; unbalanced quotes as a soft warning, because a
 * block scalar can legitimately contain one.
 * @param {string} text
 * @returns {{errors:string[], warnings:string[]}}
 */
export function yamlSanity(text) {
  const errors = [];
  const warnings = [];
  const lines = String(text ?? '').split(/\r?\n/);
  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];
    if (/^[ ]*\t/.test(line)) errors.push(`line ${i + 1}: tab used for indentation`);
    const quotes = (line.match(/"/g) || []).length;
    if (quotes % 2 === 1 && !/^\s*#/.test(line)) warnings.push(`line ${i + 1}: odd number of double quotes`);
  }
  return { errors, warnings };
}

/**
 * @param {object} ctx spec 5.6 stage context
 * @returns {Promise<object>} stage result
 */
export async function run({ wt, plan, bands, candidate }) {
  const started = Date.now();
  const reasons = [];
  const hits = [];
  const artifacts = {};

  const diff = candidate?.diffText ?? (await wt.diff()) ?? '';
  if (String(diff).trim().length === 0) {
    return {
      decision: 'bad',
      reasons: ['the candidate produced an empty diff'],
      hits: [{ key: 'patch_lines', value: 0, band: 'bad' }],
      metrics: { patch_lines: 0, files_touched: 0, added: 0, deleted: 0 },
      artifacts: {},
      costUsd: 0,
      latencyMs: Date.now() - started,
    };
  }

  const files = parseUnifiedDiff(diff);
  const stats = diffStats(diff);
  if (files.length === 0) {
    return {
      decision: 'bad',
      reasons: ['the diff is non-empty but no file headers parsed'],
      hits: [{ key: 'parse', value: 0, band: 'bad' }],
      metrics: { patch_lines: 0, files_touched: 0, added: 0, deleted: 0 },
      artifacts: { head: String(diff).slice(0, 512) },
      costUsd: 0,
      latencyMs: Date.now() - started,
    };
  }

  const paths = files.map((f) => f.path || f.oldPath).filter(Boolean);
  artifacts.files = paths.slice(0, MAX_ARTIFACT_ITEMS);

  // --- binary files
  const binaries = files.filter((f) => f.binary).map((f) => f.path || f.oldPath);
  if (binaries.length > 0) {
    reasons.push(`patch contains ${binaries.length} binary file(s): ${binaries.slice(0, 5).join(', ')}`);
    hits.push({ key: 'binary_files', value: binaries.length, band: 'bad' });
    artifacts.binaries = binaries.slice(0, MAX_ARTIFACT_ITEMS);
  }

  // --- forbidden paths
  const forbiddenGlobs = [...(Array.isArray(plan?.forbidden_paths) ? plan.forbidden_paths : []), ...DEFAULT_FORBIDDEN];
  const forbidden = [];
  for (const p of paths) {
    const pattern = matchGlobs(p, forbiddenGlobs);
    if (pattern) forbidden.push({ path: p, pattern });
  }
  if (forbidden.length > 0) {
    reasons.push(`patch touches forbidden path(s): ${forbidden.slice(0, 5).map((f) => `${f.path} (${f.pattern})`).join(', ')}`);
    hits.push({ key: 'forbidden_paths', value: forbidden.length, band: 'bad' });
    artifacts.forbidden = forbidden.slice(0, MAX_ARTIFACT_ITEMS);
  }

  // --- secrets + high entropy in ADDED lines only
  const secrets = [];
  const todos = [];
  for (const file of files) {
    const filePath = file.path || file.oldPath || '';
    const skipEntropy = matchGlobs(filePath, ENTROPY_SKIP) !== null || filePath.includes('fixtures/');
    for (const { line, text } of addedLines(file)) {
      for (const { kind, re } of SECRET_PATTERNS) {
        const m = re.exec(text);
        if (m) secrets.push({ path: filePath, line, kind, preview: redact(m[0]) });
      }
      if (TODO_RE.test(text)) todos.push({ path: filePath, line, text: text.trim().slice(0, 120) });
      if (skipEntropy) continue;
      for (const token of text.match(ENTROPY_TOKEN_RE) || []) {
        // Pure hex maxes out at 4 bits/char, so lockfile shas can never trip this.
        if (shannonEntropy(token) > ENTROPY_BITS) {
          secrets.push({ path: filePath, line, kind: 'high-entropy', preview: redact(token) });
        }
      }
    }
  }
  if (secrets.length > 0) {
    const kinds = [...new Set(secrets.map((s) => s.kind))];
    reasons.push(`patch adds ${secrets.length} possible secret(s) [${kinds.join(', ')}]`);
    hits.push({ key: 'secrets', value: secrets.length, band: 'bad' });
    artifacts.secrets = secrets.slice(0, MAX_ARTIFACT_ITEMS);
  }
  if (todos.length > 0) {
    reasons.push(`patch adds ${todos.length} TODO/placeholder marker(s)`);
    hits.push({ key: 'todo_markers', value: todos.length, band: 'unsure' });
    artifacts.todos = todos.slice(0, MAX_ARTIFACT_ITEMS);
  }

  // --- changed JSON must parse; YAML gets a sanity check (no yaml dependency)
  const unparsable = [];
  const yamlWarnings = [];
  for (const file of files) {
    if (file.status === 'D' || file.binary) continue;
    const filePath = file.path;
    if (!filePath) continue;
    const ext = extOf(filePath);
    const isJson = TEXT_EXT_JSON.includes(ext);
    const isYaml = TEXT_EXT_YAML.includes(ext);
    if (!isJson && !isYaml) continue;
    let text = null;
    try {
      text = await wt.readFile(filePath);
    } catch {
      text = null;
    }
    if (typeof text !== 'string') continue;
    if (isJson) {
      try {
        JSON.parse(text);
      } catch (err) {
        unparsable.push({ path: filePath, error: err.message.slice(0, 160) });
      }
    } else {
      const { errors, warnings } = yamlSanity(text);
      if (errors.length > 0) unparsable.push({ path: filePath, error: errors[0] });
      else if (warnings.length > 0) yamlWarnings.push({ path: filePath, warning: warnings[0] });
    }
  }
  if (unparsable.length > 0) {
    reasons.push(`changed data file(s) do not parse: ${unparsable.slice(0, 3).map((u) => u.path).join(', ')}`);
    hits.push({ key: 'unparsable_files', value: unparsable.length, band: 'bad' });
    artifacts.unparsable = unparsable.slice(0, MAX_ARTIFACT_ITEMS);
  }
  if (yamlWarnings.length > 0) {
    reasons.push(`YAML sanity warning(s): ${yamlWarnings.slice(0, 3).map((w) => w.path).join(', ')}`);
    hits.push({ key: 'yaml_warnings', value: yamlWarnings.length, band: 'unsure' });
    artifacts.yamlWarnings = yamlWarnings.slice(0, MAX_ARTIFACT_ITEMS);
  }

  // --- size bands
  const sizeHits = [
    bandFor(bands, id, 'patch_lines', stats.patchLines),
    bandFor(bands, id, 'files_touched', stats.files),
  ];
  for (const hit of sizeHits) {
    hits.push(hit);
    if (hit.band !== 'good') reasons.push(`${hit.key} = ${hit.value} is ${hit.band}`);
  }

  return {
    decision: worst(hits),
    reasons,
    hits,
    metrics: {
      patch_lines: stats.patchLines,
      files_touched: stats.files,
      added: stats.added,
      deleted: stats.deleted,
      binaries: binaries.length,
      secrets: secrets.length,
      todos: todos.length,
      forbidden: forbidden.length,
      unparsable: unparsable.length,
    },
    artifacts,
    costUsd: 0,
    latencyMs: Date.now() - started,
  };
}

export default { id, level, version, workers, gate, run };
