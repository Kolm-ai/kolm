// kolm code — in-memory worktree driver for tests (owner B, spec section 4).
//
// Scripting model (owners C and D script these two knobs):
//
//   createFakeDriver({
//     files: { 'src/a.js': 'contents\n', 'package.json': '{}\n' },   // the base checkout
//     exec: {
//       'npm test --silent': { stdout: 'ok', stderr: '', exitCode: 0 },
//       'npx tsc --noEmit -p .': (cmd, wt) => ({ exitCode: wt.readFileSync('x') ? 0 : 1 }),
//     },
//     baseSha: '0'.repeat(40),
//   })
//
// * `files` is a plain object `path -> string`. Every worktree gets its own
//   clone of it, so candidates never see each other's edits. Paths are POSIX
//   relative paths; `..` and absolute paths are rejected the same way the local
//   driver rejects them (`LadderError('path-escape')`).
// * `exec` is a table keyed by the EXACT command string passed to `wt.exec`.
//   A value may be a result object (`{stdout, stderr, exitCode, durationMs,
//   timedOut, truncated}` — every field optional) or a function
//   `(cmd, wt) => result`. An unscripted command returns
//   `{ exitCode: 127, stderr: 'fake: no script for <cmd>' }` so a missing script
//   fails loud instead of silently passing.
//
// Recorded for assertions: `driver.created`, `driver.destroyed`,
// `driver.commands` (all worktrees) and `wt.commands` (this worktree).

import { createHash } from 'node:crypto';
import { LadderError } from '../errors.js';

const NO_NEWLINE = '\\ No newline at end of file';
const DEFAULT_BASE_SHA = '0000000000000000000000000000000000000000';

/* ---------------------------------------------------------------- helpers */

function normalizeRel(rel) {
  if (typeof rel !== 'string' || rel.length === 0) {
    throw new LadderError('path-escape', 'path must be a non-empty relative string', { path: rel });
  }
  if (rel.includes('\0')) {
    throw new LadderError('path-escape', 'path must not contain NUL', { path: rel });
  }
  const posix = rel.replace(/\\/g, '/');
  if (posix.startsWith('/') || /^[A-Za-z]:/.test(posix)) {
    throw new LadderError('path-escape', `absolute paths are not allowed: ${rel}`, { path: rel });
  }
  const segments = [];
  for (const seg of posix.split('/')) {
    if (seg === '' || seg === '.') continue;
    if (seg === '..') {
      throw new LadderError('path-escape', `path escapes the worktree: ${rel}`, { path: rel });
    }
    segments.push(seg);
  }
  if (segments.length === 0) {
    throw new LadderError('path-escape', `path escapes the worktree: ${rel}`, { path: rel });
  }
  return segments.join('/');
}

/** Minimal `*` / `**` / `?` matcher (POSIX paths, `/` is a segment boundary). */
export function globToRegExp(glob) {
  let source = '';
  for (let i = 0; i < glob.length; i += 1) {
    const c = glob[i];
    if (c === '*') {
      if (glob[i + 1] === '*') {
        if (glob[i + 2] === '/') {
          source += '(?:.*/)?';
          i += 2;
        } else {
          source += '.*';
          i += 1;
        }
      } else {
        source += '[^/]*';
      }
    } else if (c === '?') {
      source += '[^/]';
    } else if ('.+^${}()|[]\\'.includes(c)) {
      source += `\\${c}`;
    } else {
      source += c;
    }
  }
  return new RegExp(`^${source}$`);
}

function matchesGlob(pathStr, glob) {
  if (!glob) return true;
  return globToRegExp(glob).test(pathStr);
}

function blobSha(content) {
  const buf = Buffer.from(content, 'utf8');
  return createHash('sha1')
    .update(`blob ${buf.length}\0`)
    .update(buf)
    .digest('hex');
}

function shortSha(content) {
  return content === null ? '0000000' : blobSha(content).slice(0, 7);
}

function splitContent(content) {
  if (content === '' || content === null || content === undefined) {
    return { lines: [], eofNewline: true };
  }
  const eofNewline = content.endsWith('\n');
  const body = eofNewline ? content.slice(0, -1) : content;
  return { lines: body.split('\n'), eofNewline };
}

function joinContent(lines, eofNewline) {
  if (lines.length === 0) return '';
  return lines.join('\n') + (eofNewline ? '\n' : '');
}

/* ------------------------------------------------------------ diff writer */

/**
 * Unified diff for one file. Modified files get a single whole-file hunk
 * (all removed lines, then all added lines) with correct `@@` counts, so a
 * standard unified-diff parser reads the right file/line totals out of it.
 */
export function buildFileDiff(pathStr, oldContent, newContent) {
  if (oldContent === newContent) return '';
  const isNew = oldContent === null || oldContent === undefined;
  const isDeleted = newContent === null || newContent === undefined;
  const oldSide = splitContent(isNew ? '' : oldContent);
  const newSide = splitContent(isDeleted ? '' : newContent);

  const out = [`diff --git a/${pathStr} b/${pathStr}`];
  if (isNew) out.push('new file mode 100644');
  if (isDeleted) out.push('deleted file mode 100644');
  out.push(
    `index ${shortSha(isNew ? null : oldContent)}..${shortSha(isDeleted ? null : newContent)}${isNew || isDeleted ? '' : ' 100644'}`,
  );
  out.push(isNew ? '--- /dev/null' : `--- a/${pathStr}`);
  out.push(isDeleted ? '+++ /dev/null' : `+++ b/${pathStr}`);

  const oldStart = oldSide.lines.length === 0 ? 0 : 1;
  const newStart = newSide.lines.length === 0 ? 0 : 1;
  out.push(`@@ -${oldStart},${oldSide.lines.length} +${newStart},${newSide.lines.length} @@`);

  for (const line of oldSide.lines) out.push(`-${line}`);
  if (oldSide.lines.length > 0 && !oldSide.eofNewline) out.push(NO_NEWLINE);
  for (const line of newSide.lines) out.push(`+${line}`);
  if (newSide.lines.length > 0 && !newSide.eofNewline) out.push(NO_NEWLINE);

  return `${out.join('\n')}\n`;
}

/* ------------------------------------------------------------ diff parser */

/** Parse the subset of unified diff this driver emits (and simple patches). */
export function parseUnifiedDiff(diff) {
  const lines = String(diff).split('\n');
  const files = [];
  let current = null;
  let hunk = null;

  const closeHunk = () => {
    if (hunk && current) current.hunks.push(hunk);
    hunk = null;
  };

  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];
    if (line.startsWith('diff --git ')) {
      closeHunk();
      current = { path: null, isNew: false, isDeleted: false, hunks: [] };
      files.push(current);
      const match = /^diff --git a\/(.*) b\/(.*)$/.exec(line);
      if (match) current.path = match[2];
      continue;
    }
    if (!current) continue;
    if (line.startsWith('new file mode')) { current.isNew = true; continue; }
    if (line.startsWith('deleted file mode')) { current.isDeleted = true; continue; }
    if (line.startsWith('--- ')) {
      closeHunk();
      const from = line.slice(4).trim();
      if (from === '/dev/null') current.isNew = true;
      continue;
    }
    if (line.startsWith('+++ ')) {
      closeHunk();
      const to = line.slice(4).trim();
      if (to === '/dev/null') current.isDeleted = true;
      else if (!current.path) current.path = to.replace(/^b\//, '');
      continue;
    }
    if (line.startsWith('@@')) {
      closeHunk();
      const match = /^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@/.exec(line);
      if (!match) continue;
      hunk = {
        oldStart: Number(match[1]),
        oldCount: match[2] === undefined ? 1 : Number(match[2]),
        newStart: Number(match[3]),
        newCount: match[4] === undefined ? 1 : Number(match[4]),
        lines: [],
      };
      continue;
    }
    if (!hunk) continue;
    if (line === NO_NEWLINE) {
      const last = hunk.lines[hunk.lines.length - 1];
      if (last) last.noNewline = true;
      continue;
    }
    if (line === '' && i === lines.length - 1) continue;
    const op = line.charAt(0);
    if (op === ' ' || op === '-' || op === '+') {
      hunk.lines.push({ op, text: line.slice(1), noNewline: false });
    } else if (line === '') {
      hunk.lines.push({ op: ' ', text: '', noNewline: false });
    } else {
      // Anything else ends the hunk body (index/mode trailers, next header).
      closeHunk();
    }
  }
  closeHunk();
  return files.filter((f) => f.path);
}

function applyHunks(originalLines, originalEof, hunks) {
  const out = [];
  let cursor = 0;
  let eof = originalEof;
  for (const h of hunks) {
    const start = h.oldStart > 0 ? h.oldStart - 1 : 0;
    if (start < cursor) return { ok: false, error: 'overlapping hunks' };
    if (start > originalLines.length) return { ok: false, error: 'hunk starts past end of file' };
    while (cursor < start) { out.push(originalLines[cursor]); cursor += 1; }
    let lastNewSide = null;
    for (const line of h.lines) {
      if (line.op === ' ') {
        if (originalLines[cursor] !== line.text) {
          return { ok: false, error: `context mismatch at line ${cursor + 1}` };
        }
        out.push(originalLines[cursor]);
        cursor += 1;
        lastNewSide = line;
      } else if (line.op === '-') {
        if (originalLines[cursor] !== line.text) {
          return { ok: false, error: `removed line mismatch at line ${cursor + 1}` };
        }
        cursor += 1;
      } else {
        out.push(line.text);
        lastNewSide = line;
      }
    }
    if (cursor >= originalLines.length && lastNewSide) eof = !lastNewSide.noNewline;
  }
  while (cursor < originalLines.length) { out.push(originalLines[cursor]); cursor += 1; }
  return { ok: true, lines: out, eofNewline: eof };
}

/* ------------------------------------------------------------------ driver */

/**
 * @param {object} [spec]
 * @param {Record<string,string>} [spec.files] base checkout, path -> content
 * @param {Record<string,object|Function>} [spec.exec] scripted command table
 * @param {string} [spec.baseSha]
 */
export function createFakeDriver({ files = {}, exec = {}, baseSha = DEFAULT_BASE_SHA } = {}) {
  const baseFiles = new Map();
  for (const [k, v] of Object.entries(files)) baseFiles.set(normalizeRel(k), String(v));

  const driver = {
    kind: 'fake',
    baseSha,
    created: [],
    destroyed: [],
    commands: [],
    worktrees: new Map(),
    create,
    open,
    destroyRun,
  };

  function key(runId, candidateId) {
    return `${runId}/${candidateId}`;
  }

  function makeWorktree({ repoPath = null, repoUrl = null, runId, candidateId }) {
    const base = new Map(baseFiles);
    const current = new Map(baseFiles);
    const branch = `kolm/code/${runId}/${candidateId}`;
    let attempt = 0;

    const wt = {
      id: candidateId,
      kind: 'fake',
      path: `/fake/${runId}/${candidateId}`,
      repoPath,
      repoUrl,
      runId,
      candidateId,
      branch,
      commands: [],
      /** Test-only view of the in-memory tree. */
      files: current,
      baseFiles: base,

      async baseSha() {
        return baseSha;
      },

      /** Synchronous peek used by scripted `exec` functions. */
      readFileSync(rel) {
        return current.has(normalizeRel(rel)) ? current.get(normalizeRel(rel)) : null;
      },

      async exec(cmd, execOpts = {}) {
        if (typeof cmd !== 'string' || cmd.trim() === '') {
          throw new LadderError('bad-config', 'exec requires a non-empty command string', { cmd });
        }
        const scripted = Object.prototype.hasOwnProperty.call(exec, cmd) ? exec[cmd] : undefined;
        let raw;
        if (typeof scripted === 'function') raw = scripted(cmd, wt, execOpts);
        else if (scripted && typeof scripted === 'object') raw = scripted;
        else raw = { exitCode: 127, stderr: `fake: no script for ${cmd}` };
        raw = await raw;
        const result = {
          stdout: raw?.stdout ?? '',
          stderr: raw?.stderr ?? '',
          exitCode: raw?.exitCode ?? 0,
          durationMs: raw?.durationMs ?? 0,
          timedOut: raw?.timedOut ?? false,
          truncated: raw?.truncated ?? false,
        };
        const record = {
          cmd,
          cwd: execOpts.cwd ?? null,
          timeoutMs: execOpts.timeoutMs ?? null,
          env: execOpts.env ?? null,
          runId,
          candidateId,
          result,
        };
        wt.commands.push(record);
        driver.commands.push(record);
        return result;
      },

      async readFile(rel) {
        const p = normalizeRel(rel);
        return current.has(p) ? current.get(p) : null;
      },

      async writeFile(rel, content) {
        const p = normalizeRel(rel);
        current.set(p, String(content ?? ''));
        return { path: p };
      },

      async deleteFile(rel) {
        const p = normalizeRel(rel);
        current.delete(p);
        return { path: p };
      },

      async listFiles({ glob, limit = 5000 } = {}) {
        const out = [];
        for (const p of [...current.keys()].sort()) {
          if (out.length >= limit) break;
          if (matchesGlob(p, glob)) out.push(p);
        }
        return out;
      },

      async search(pattern, { glob, limit = 200 } = {}) {
        if (typeof pattern !== 'string' || pattern.length === 0) {
          throw new LadderError('bad-config', 'search requires a non-empty pattern', { pattern });
        }
        let re;
        try {
          re = new RegExp(pattern);
        } catch (err) {
          throw new LadderError('bad-config', `invalid search pattern: ${err.message}`, { pattern });
        }
        const hits = [];
        for (const p of [...current.keys()].sort()) {
          if (hits.length >= limit) break;
          if (!matchesGlob(p, glob)) continue;
          const { lines } = splitContent(current.get(p));
          for (let i = 0; i < lines.length; i += 1) {
            if (hits.length >= limit) break;
            if (re.test(lines[i])) hits.push({ path: p, line: i + 1, text: lines[i] });
          }
        }
        return hits;
      },

      async diff() {
        const paths = [...new Set([...base.keys(), ...current.keys()])].sort();
        let out = '';
        for (const p of paths) {
          const oldContent = base.has(p) ? base.get(p) : null;
          const newContent = current.has(p) ? current.get(p) : null;
          out += buildFileDiff(p, oldContent, newContent);
        }
        return out;
      },

      async changedFiles() {
        const paths = [...new Set([...base.keys(), ...current.keys()])].sort();
        const rows = [];
        for (const p of paths) {
          const inBase = base.has(p);
          const inCurrent = current.has(p);
          if (inBase && inCurrent && base.get(p) === current.get(p)) continue;
          const oldLines = inBase ? splitContent(base.get(p)).lines.length : 0;
          const newLines = inCurrent ? splitContent(current.get(p)).lines.length : 0;
          if (!inBase) rows.push({ path: p, status: 'A', added: newLines, deleted: 0 });
          else if (!inCurrent) rows.push({ path: p, status: 'D', added: 0, deleted: oldLines });
          else rows.push({ path: p, status: 'M', added: newLines, deleted: oldLines });
        }
        return rows;
      },

      async applyPatch(diff) {
        if (typeof diff !== 'string' || diff.trim() === '') {
          return { ok: false, error: 'empty patch' };
        }
        let parsed;
        try {
          parsed = parseUnifiedDiff(diff);
        } catch (err) {
          return { ok: false, error: `unparsable patch: ${err.message}` };
        }
        if (parsed.length === 0) return { ok: false, error: 'patch contains no file sections' };
        const staged = new Map();
        const removed = new Set();
        for (const file of parsed) {
          let p;
          try {
            p = normalizeRel(file.path);
          } catch (err) {
            return { ok: false, error: err.message };
          }
          if (file.isDeleted) {
            if (!current.has(p)) return { ok: false, error: `cannot delete missing file ${p}` };
            removed.add(p);
            continue;
          }
          const existing = current.has(p) ? current.get(p) : null;
          if (file.isNew && existing !== null) {
            return { ok: false, error: `file already exists: ${p}` };
          }
          const src = splitContent(existing ?? '');
          const applied = applyHunks(src.lines, existing === null ? true : src.eofNewline, file.hunks);
          if (!applied.ok) return { ok: false, error: `${p}: ${applied.error}` };
          staged.set(p, joinContent(applied.lines, applied.eofNewline));
        }
        for (const p of removed) current.delete(p);
        for (const [p, content] of staged) current.set(p, content);
        return { ok: true };
      },

      async commit(message) {
        const sha = createHash('sha1').update(`${runId}${candidateId}${attempt}`).digest('hex');
        attempt += 1;
        wt.lastCommitMessage = String(message ?? '');
        return { sha, branch };
      },

      async destroy() {
        driver.destroyed.push({ runId, candidateId, id: candidateId });
        driver.worktrees.delete(key(runId, candidateId));
        return { removed: true };
      },
    };
    return wt;
  }

  async function create({ repoPath, repoUrl, runId, candidateId } = {}) {
    if (typeof runId !== 'string' || typeof candidateId !== 'string' || !runId || !candidateId) {
      throw new LadderError('bad-config', 'runId and candidateId are required', { runId, candidateId });
    }
    const wt = makeWorktree({ repoPath, repoUrl, runId, candidateId });
    driver.worktrees.set(key(runId, candidateId), wt);
    driver.created.push({ runId, candidateId, id: candidateId, path: wt.path });
    return wt;
  }

  async function open({ repoPath, repoUrl, runId, candidateId } = {}) {
    const existing = driver.worktrees.get(key(runId, candidateId));
    if (existing) return existing;
    return create({ repoPath, repoUrl, runId, candidateId });
  }

  async function destroyRun(runId) {
    let removed = 0;
    for (const [k, wt] of [...driver.worktrees.entries()]) {
      if (!k.startsWith(`${runId}/`)) continue;
      await wt.destroy();
      removed += 1;
    }
    return { removed };
  }

  return driver;
}

export default createFakeDriver;
