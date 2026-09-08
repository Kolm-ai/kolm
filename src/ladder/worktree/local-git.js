// kolm code — local `git worktree` driver (owner B, spec section 4).
//
// One throwaway detached worktree per candidate under
// `<worktreesDir>/<runId>/<candidateId>`. Metadata (repo path, base sha) is
// stored in a sibling `<candidateId>.kolm.json` file so `open()` can re-attach
// after a resume WITHOUT writing anything inside the worktree itself (an extra
// file there would show up in `listFiles`, `diff` and `changedFiles`).
//
// Process rules: git is spawned with an args array and `shell:false` so no user
// string is ever parsed by a shell. Only `wt.exec(cmd)` — which takes an
// arbitrary user command — uses `shell:true`.

import { spawn } from 'node:child_process';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { LadderError } from '../errors.js';

export const DEFAULT_TIMEOUT_MS = 300000;
export const DEFAULT_MAX_OUTPUT_BYTES = 1 << 20;

const GIT_IDENTITY = ['-c', 'user.name=kolm code', '-c', 'user.email=code@kolm.ai'];
const SEGMENT_RE = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;
const IS_WINDOWS = process.platform === 'win32';

/* ------------------------------------------------------------------ process */

/**
 * Spawn a process, capture capped output, kill the whole tree on timeout.
 * Never throws for a non-zero exit; a spawn failure is reported as exitCode -1.
 *
 * `maxOutputBytes` is applied per stream (stdout and stderr each get the cap);
 * `truncated` is true when either stream hit it.
 *
 * @returns {Promise<{stdout:string, stderr:string, exitCode:number,
 *   durationMs:number, timedOut:boolean, truncated:boolean}>}
 */
export function runProcess(file, args = [], options = {}) {
  const {
    cwd,
    env,
    shell = false,
    timeoutMs = DEFAULT_TIMEOUT_MS,
    maxOutputBytes = DEFAULT_MAX_OUTPUT_BYTES,
  } = options;
  const started = Date.now();

  return new Promise((resolve) => {
    let settled = false;
    let timedOut = false;
    let truncated = false;
    const chunks = { stdout: [], stderr: [] };
    const seen = { stdout: 0, stderr: 0 };
    let timer = null;
    let killTimer = null;

    const finish = (exitCode) => {
      if (settled) return;
      settled = true;
      if (timer) clearTimeout(timer);
      if (killTimer) clearTimeout(killTimer);
      resolve({
        stdout: Buffer.concat(chunks.stdout).toString('utf8'),
        stderr: Buffer.concat(chunks.stderr).toString('utf8'),
        exitCode,
        durationMs: Date.now() - started,
        timedOut,
        truncated,
      });
    };

    let child;
    try {
      child = spawn(file, args, {
        cwd,
        shell,
        env: env ? { ...process.env, ...env } : process.env,
        windowsHide: true,
        // POSIX: own process group so the whole tree can be signalled.
        detached: !IS_WINDOWS,
      });
    } catch (err) {
      chunks.stderr.push(Buffer.from(String(err?.message ?? err), 'utf8'));
      finish(-1);
      return;
    }

    const collect = (name) => (chunk) => {
      const remaining = maxOutputBytes - seen[name];
      if (remaining <= 0) {
        truncated = true;
        return;
      }
      if (chunk.length > remaining) {
        chunks[name].push(chunk.subarray(0, remaining));
        seen[name] = maxOutputBytes;
        truncated = true;
        return;
      }
      chunks[name].push(chunk);
      seen[name] += chunk.length;
    };

    child.stdout?.on('data', collect('stdout'));
    child.stderr?.on('data', collect('stderr'));
    child.on('error', (err) => {
      chunks.stderr.push(Buffer.from(String(err?.message ?? err), 'utf8'));
      finish(-1);
    });
    child.on('close', (code, signal) => {
      finish(typeof code === 'number' ? code : (signal ? 124 : -1));
    });

    if (timeoutMs > 0) {
      timer = setTimeout(() => {
        timedOut = true;
        killProcessTree(child);
        // If the tree refuses to die, still return within a bounded window.
        killTimer = setTimeout(() => finish(124), 3000);
        killTimer.unref?.();
      }, timeoutMs);
      timer.unref?.();
    }
  });
}

function killProcessTree(child) {
  if (!child || child.pid === undefined) return;
  if (IS_WINDOWS) {
    try {
      const killer = spawn('taskkill', ['/pid', String(child.pid), '/T', '/F'], {
        windowsHide: true,
        stdio: 'ignore',
      });
      killer.on('error', () => {});
      killer.unref?.();
    } catch {
      try { child.kill('SIGKILL'); } catch { /* already gone */ }
    }
    return;
  }
  try {
    process.kill(-child.pid, 'SIGKILL');
  } catch {
    try { child.kill('SIGKILL'); } catch { /* already gone */ }
  }
}

/* ---------------------------------------------------------------- utilities */

function toPosix(p) {
  return String(p).replace(/\\/g, '/');
}

function assertSegment(value, field) {
  if (typeof value !== 'string' || !SEGMENT_RE.test(value)) {
    throw new LadderError(
      'bad-config',
      `${field} must be a single safe path segment (letters, digits, . _ -), got ${JSON.stringify(value)}`,
      { field, value },
    );
  }
  return value;
}

/**
 * Resolve `rel` inside `root`, refusing absolute paths, `..` segments and any
 * result that lands outside the root.
 */
export function resolveInside(root, rel) {
  if (typeof rel !== 'string' || rel.length === 0) {
    throw new LadderError('path-escape', 'path must be a non-empty relative string', { path: rel });
  }
  if (rel.includes('\0')) {
    throw new LadderError('path-escape', 'path must not contain NUL', { path: rel });
  }
  const normalized = toPosix(rel);
  if (normalized.startsWith('/') || /^[A-Za-z]:/.test(normalized) || path.isAbsolute(rel)) {
    throw new LadderError('path-escape', `absolute paths are not allowed: ${rel}`, { path: rel });
  }
  if (normalized.split('/').some((seg) => seg === '..')) {
    throw new LadderError('path-escape', `path escapes the worktree: ${rel}`, { path: rel });
  }
  const rootResolved = path.resolve(root);
  const resolved = path.resolve(rootResolved, normalized);
  if (resolved !== rootResolved && !resolved.startsWith(rootResolved + path.sep)) {
    throw new LadderError('path-escape', `path escapes the worktree: ${rel}`, { path: rel });
  }
  return resolved;
}

function splitLines(text) {
  if (!text) return [];
  const out = text.split('\n');
  if (out.length && out[out.length - 1] === '') out.pop();
  return out.map((l) => (l.endsWith('\r') ? l.slice(0, -1) : l));
}

function defaultWorktreesDir() {
  return path.join(os.homedir() || os.tmpdir(), '.kolm', 'code', 'worktrees');
}

async function dirHasContent(dir) {
  try {
    const entries = await fs.readdir(dir);
    return entries.length > 0;
  } catch {
    return false;
  }
}

async function isBinaryFile(abs) {
  let handle;
  try {
    handle = await fs.open(abs, 'r');
    const buf = Buffer.alloc(8000);
    const { bytesRead } = await handle.read(buf, 0, 8000, 0);
    return buf.subarray(0, bytesRead).includes(0);
  } catch {
    return false;
  } finally {
    await handle?.close().catch(() => {});
  }
}

async function countLines(abs) {
  try {
    const text = await fs.readFile(abs, 'utf8');
    if (text.length === 0) return 0;
    return text.endsWith('\n') ? text.split('\n').length - 1 : text.split('\n').length;
  } catch {
    return 0;
  }
}

/* ------------------------------------------------------------------- driver */

/**
 * @param {object} config  ladder config; only `worktreesDir` and
 *                         `stageTimeoutMs` are read.
 * @param {object} [opts]  `{ worktreesDir, timeoutMs, maxOutputBytes, reuse }`
 */
export function createLocalGitDriver(config = {}, opts = {}) {
  const worktreesDir = path.resolve(
    opts.worktreesDir ?? config.worktreesDir ?? defaultWorktreesDir(),
  );
  const defaultTimeoutMs = opts.timeoutMs ?? config.stageTimeoutMs ?? DEFAULT_TIMEOUT_MS;
  const defaultMaxOutputBytes = opts.maxOutputBytes ?? DEFAULT_MAX_OUTPUT_BYTES;
  const reuseByDefault = Boolean(opts.reuse);

  const git = (args, gitOpts = {}) =>
    runProcess('git', args, {
      shell: false,
      timeoutMs: gitOpts.timeoutMs ?? defaultTimeoutMs,
      maxOutputBytes: gitOpts.maxOutputBytes ?? defaultMaxOutputBytes,
      cwd: gitOpts.cwd,
      env: gitOpts.env,
    });

  async function gitOrThrow(args, gitOpts, code, what) {
    const res = await git(args, gitOpts);
    if (res.exitCode !== 0) {
      throw new LadderError(code, `${what}: ${(res.stderr || res.stdout).trim() || `git exited ${res.exitCode}`}`, {
        args,
        cwd: gitOpts?.cwd,
        exitCode: res.exitCode,
        stderr: res.stderr,
      });
    }
    return res;
  }

  async function repoRoot(repoPath) {
    if (typeof repoPath !== 'string' || repoPath.length === 0) {
      throw new LadderError('not-a-git-repo', 'repoPath is required', { repoPath });
    }
    const res = await git(['rev-parse', '--show-toplevel'], { cwd: repoPath, timeoutMs: 60000 });
    if (res.exitCode !== 0) {
      throw new LadderError('not-a-git-repo', `${repoPath} is not a git repository`, {
        repoPath,
        stderr: res.stderr.trim(),
      });
    }
    return path.resolve(res.stdout.trim());
  }

  const metaPathFor = (runId, candidateId) =>
    path.join(worktreesDir, runId, `${candidateId}.kolm.json`);
  const worktreePathFor = (runId, candidateId) => path.join(worktreesDir, runId, candidateId);

  async function readMeta(runId, candidateId) {
    try {
      return JSON.parse(await fs.readFile(metaPathFor(runId, candidateId), 'utf8'));
    } catch {
      return null;
    }
  }

  /* ------------------------------------------------------------ worktree */

  function makeWorktree({ repoPath, runId, candidateId, wtPath, baseSha }) {
    const root = path.resolve(wtPath);
    const branch = `kolm/code/${runId}/${candidateId}`;

    const wtGit = (args, o = {}) => git(args, { cwd: root, ...o });

    async function untrackedFiles() {
      const res = await wtGit(['ls-files', '--others', '--exclude-standard', '-z']);
      if (res.exitCode !== 0) return [];
      return res.stdout.split('\0').filter(Boolean).map(toPosix);
    }

    const wt = {
      id: candidateId,
      kind: 'local',
      path: root,
      repoPath,
      runId,
      candidateId,
      branch,

      async baseSha() {
        return baseSha;
      },

      async exec(cmd, execOpts = {}) {
        if (typeof cmd !== 'string' || cmd.trim() === '') {
          throw new LadderError('bad-config', 'exec requires a non-empty command string', { cmd });
        }
        const cwd = execOpts.cwd ? resolveInside(root, execOpts.cwd) : root;
        // Arbitrary user command -> a shell is required here (and only here).
        return runProcess(cmd, [], {
          cwd,
          shell: true,
          env: execOpts.env,
          timeoutMs: execOpts.timeoutMs ?? DEFAULT_TIMEOUT_MS,
          maxOutputBytes: execOpts.maxOutputBytes ?? DEFAULT_MAX_OUTPUT_BYTES,
        });
      },

      async readFile(rel) {
        const abs = resolveInside(root, rel);
        try {
          return await fs.readFile(abs, 'utf8');
        } catch (err) {
          if (err && (err.code === 'ENOENT' || err.code === 'EISDIR' || err.code === 'ENOTDIR')) return null;
          throw err;
        }
      },

      async writeFile(rel, content) {
        const abs = resolveInside(root, rel);
        await fs.mkdir(path.dirname(abs), { recursive: true });
        await fs.writeFile(abs, content ?? '', 'utf8');
        return { path: toPosix(rel) };
      },

      async deleteFile(rel) {
        const abs = resolveInside(root, rel);
        await fs.rm(abs, { force: true, recursive: false });
        return { path: toPosix(rel) };
      },

      async listFiles({ glob, limit = 5000 } = {}) {
        const args = ['ls-files', '--cached', '--others', '--exclude-standard', '-z'];
        if (glob) args.push('--', glob);
        const res = await wtGit(args);
        if (res.exitCode !== 0) return [];
        const seenPaths = new Set();
        const out = [];
        for (const raw of res.stdout.split('\0')) {
          if (!raw) continue;
          const p = toPosix(raw);
          if (seenPaths.has(p)) continue;
          seenPaths.add(p);
          out.push(p);
        }
        out.sort();
        // `--cached` still lists files deleted from the working tree; drop them.
        const alive = [];
        for (const p of out) {
          if (alive.length >= limit) break;
          try {
            await fs.stat(path.resolve(root, p));
            alive.push(p);
          } catch { /* deleted in the working tree */ }
        }
        return alive;
      },

      async search(pattern, { glob, limit = 200 } = {}) {
        if (typeof pattern !== 'string' || pattern.length === 0) {
          throw new LadderError('bad-config', 'search requires a non-empty pattern', { pattern });
        }
        const args = ['-c', 'core.quotepath=false', 'grep', '-n', '-I', '--untracked', '-e', pattern];
        if (glob) args.push('--', glob);
        const res = await wtGit(args);
        // git grep exits 1 when nothing matched; that is not an error.
        if (res.exitCode === 1) return [];
        if (res.exitCode !== 0) {
          throw new LadderError('git-failed', `git grep failed: ${res.stderr.trim()}`, {
            exitCode: res.exitCode,
            stderr: res.stderr,
          });
        }
        const hits = [];
        for (const line of splitLines(res.stdout)) {
          if (hits.length >= limit) break;
          const first = line.indexOf(':');
          if (first < 0) continue;
          const second = line.indexOf(':', first + 1);
          if (second < 0) continue;
          const lineNo = Number(line.slice(first + 1, second));
          if (!Number.isFinite(lineNo)) continue;
          hits.push({ path: toPosix(line.slice(0, first)), line: lineNo, text: line.slice(second + 1) });
        }
        return hits;
      },

      async diff() {
        const tracked = await wtGit([
          '-c', 'core.quotepath=false',
          'diff', '--no-ext-diff', '--no-color', '--no-renames', baseSha, '--',
        ], { maxOutputBytes: 32 << 20 });
        if (tracked.exitCode !== 0 && tracked.exitCode !== 1) {
          throw new LadderError('git-failed', `git diff failed: ${tracked.stderr.trim()}`, {
            exitCode: tracked.exitCode,
            stderr: tracked.stderr,
          });
        }
        const parts = [];
        if (tracked.stdout.trim()) parts.push(ensureTrailingNewline(tracked.stdout));

        for (const rel of await untrackedFiles()) {
          // `--no-index` against /dev/null is understood by git on Windows too
          // (verified on this machine); it yields a proper `new file mode` hunk.
          const one = await wtGit([
            '-c', 'core.quotepath=false',
            'diff', '--no-ext-diff', '--no-color', '--no-index', '--', '/dev/null', rel,
          ], { maxOutputBytes: 32 << 20 });
          if (one.stdout.trim()) parts.push(ensureTrailingNewline(one.stdout));
        }
        return parts.join('');
      },

      async changedFiles() {
        const numstat = await wtGit([
          '-c', 'core.quotepath=false',
          'diff', '--no-ext-diff', '--no-renames', '--numstat', '-z', baseSha, '--',
        ]);
        const nameStatus = await wtGit([
          '-c', 'core.quotepath=false',
          'diff', '--no-ext-diff', '--no-renames', '--name-status', '-z', baseSha, '--',
        ]);
        const statusByPath = new Map();
        {
          const fields = nameStatus.stdout.split('\0').filter((f) => f !== '');
          for (let i = 0; i + 1 < fields.length; i += 2) {
            statusByPath.set(toPosix(fields[i + 1]), fields[i].charAt(0));
          }
        }
        const rows = [];
        {
          // `--numstat -z` emits "added\tdeleted\tpath\0" records.
          for (const record of numstat.stdout.split('\0')) {
            if (!record) continue;
            const tab1 = record.indexOf('\t');
            const tab2 = record.indexOf('\t', tab1 + 1);
            if (tab1 < 0 || tab2 < 0) continue;
            const added = record.slice(0, tab1);
            const deleted = record.slice(tab1 + 1, tab2);
            const p = toPosix(record.slice(tab2 + 1));
            rows.push({
              path: p,
              status: statusByPath.get(p) ?? 'M',
              added: added === '-' ? 0 : Number(added) || 0,
              deleted: deleted === '-' ? 0 : Number(deleted) || 0,
            });
          }
        }
        for (const rel of await untrackedFiles()) {
          const abs = path.resolve(root, rel);
          const binary = await isBinaryFile(abs);
          rows.push({ path: rel, status: 'A', added: binary ? 0 : await countLines(abs), deleted: 0 });
        }
        rows.sort((a, b) => (a.path < b.path ? -1 : a.path > b.path ? 1 : 0));
        return rows;
      },

      async applyPatch(diff) {
        if (typeof diff !== 'string' || diff.trim() === '') {
          return { ok: false, error: 'empty patch' };
        }
        const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'kolm-code-patch-'));
        const file = path.join(dir, 'candidate.patch');
        try {
          await fs.writeFile(file, ensureTrailingNewline(diff), 'utf8');
          const res = await wtGit(['apply', '--3way', '--whitespace=nowarn', file]);
          if (res.exitCode === 0) return { ok: true };
          return { ok: false, error: (res.stderr || res.stdout).trim() || `git apply exited ${res.exitCode}` };
        } finally {
          await fs.rm(dir, { recursive: true, force: true }).catch(() => {});
        }
      },

      async commit(message) {
        // `git add -A` is confined to this worktree, which is a throwaway
        // candidate checkout created and owned by kolm code. This is the ONLY
        // place in the ladder that stages everything; nothing here can reach
        // the operator's own repository working tree.
        await gitOrThrow(['add', '-A'], { cwd: root }, 'git-failed', 'git add failed');
        // A resumed run may find the candidate already committed. A clean tree
        // is then not a failure: the existing HEAD is the commit.
        const status = await gitOrThrow(['status', '--porcelain'], { cwd: root }, 'git-failed', 'git status failed');
        const commitRes = status.stdout.trim() === ''
          ? { exitCode: 0, stdout: '', stderr: '' }
          : await wtGit([
            ...GIT_IDENTITY,
            'commit', '-q', '-m', String(message ?? ''), '--allow-empty-message',
          ]);
        if (commitRes.exitCode !== 0) {
          throw new LadderError(
            'commit-failed',
            `git commit failed: ${(commitRes.stderr || commitRes.stdout).trim()}`,
            { exitCode: commitRes.exitCode, stderr: commitRes.stderr, stdout: commitRes.stdout },
          );
        }
        const shaRes = await gitOrThrow(['rev-parse', 'HEAD'], { cwd: root }, 'git-failed', 'git rev-parse HEAD failed');
        const sha = shaRes.stdout.trim();
        // A branch cannot be checked out in two worktrees, so the worktree stays
        // detached and the branch ref is only pointed at the new commit.
        await gitOrThrow(['branch', '-f', branch, sha], { cwd: root }, 'git-failed', 'git branch failed');
        return { sha, branch };
      },

      async destroy() {
        const res = await git(['worktree', 'remove', '--force', root], { cwd: repoPath });
        if (res.exitCode !== 0) {
          await fs.rm(root, { recursive: true, force: true }).catch(() => {});
        }
        await git(['worktree', 'prune'], { cwd: repoPath });
        await fs.rm(metaPathFor(runId, candidateId), { force: true }).catch(() => {});
        return { removed: true };
      },
    };
    return wt;
  }

  /* -------------------------------------------------------------- create */

  async function create({
    repoPath,
    baseRef = 'HEAD',
    runId,
    candidateId,
    reuse = reuseByDefault,
  } = {}) {
    assertSegment(runId, 'runId');
    assertSegment(candidateId, 'candidateId');
    const root = await repoRoot(repoPath);

    const shaRes = await git(['rev-parse', `${baseRef}^{commit}`], { cwd: root, timeoutMs: 60000 });
    if (shaRes.exitCode !== 0) {
      throw new LadderError('bad-ref', `cannot resolve baseRef ${baseRef} in ${root}`, {
        baseRef,
        repoPath: root,
        stderr: shaRes.stderr.trim(),
      });
    }
    const baseSha = shaRes.stdout.trim();

    const wtPath = worktreePathFor(runId, candidateId);
    if (await dirHasContent(wtPath)) {
      if (!reuse) {
        throw new LadderError('worktree-exists', `a worktree already exists at ${wtPath}`, {
          path: wtPath,
          runId,
          candidateId,
        });
      }
      return open({ repoPath: root, runId, candidateId });
    }

    await fs.mkdir(path.dirname(wtPath), { recursive: true });
    await gitOrThrow(
      ['worktree', 'add', '--detach', wtPath, baseSha],
      { cwd: root },
      'worktree-failed',
      'git worktree add failed',
    );
    await fs.writeFile(
      metaPathFor(runId, candidateId),
      `${JSON.stringify({ repoPath: root, runId, candidateId, baseSha, path: path.resolve(wtPath), createdAt: new Date().toISOString() }, null, 2)}\n`,
      'utf8',
    );
    return makeWorktree({ repoPath: root, runId, candidateId, wtPath, baseSha });
  }

  /* ---------------------------------------------------------------- open */

  async function open({ repoPath, runId, candidateId } = {}) {
    assertSegment(runId, 'runId');
    assertSegment(candidateId, 'candidateId');
    const wtPath = worktreePathFor(runId, candidateId);
    if (!(await dirHasContent(wtPath))) {
      throw new LadderError('worktree-missing', `no worktree at ${wtPath}`, { path: wtPath, runId, candidateId });
    }
    const meta = await readMeta(runId, candidateId);
    const root = await repoRoot(meta?.repoPath ?? repoPath ?? wtPath);
    let baseSha = meta?.baseSha;
    if (!baseSha) {
      const res = await git(['rev-parse', 'HEAD'], { cwd: wtPath, timeoutMs: 60000 });
      if (res.exitCode !== 0) {
        throw new LadderError('worktree-missing', `cannot read HEAD of ${wtPath}`, { path: wtPath });
      }
      baseSha = res.stdout.trim();
    }
    return makeWorktree({ repoPath: root, runId, candidateId, wtPath, baseSha });
  }

  /* --------------------------------------------------------- destroyRun */

  async function destroyRun(runId, destroyOpts = {}) {
    assertSegment(runId, 'runId');
    const runDir = path.join(worktreesDir, runId);
    const repoPaths = new Set();
    if (destroyOpts.repoPath) repoPaths.add(destroyOpts.repoPath);
    let entries = [];
    try {
      entries = await fs.readdir(runDir, { withFileTypes: true });
    } catch {
      return { removed: 0 };
    }
    for (const entry of entries) {
      if (!entry.isFile() || !entry.name.endsWith('.kolm.json')) continue;
      try {
        const meta = JSON.parse(await fs.readFile(path.join(runDir, entry.name), 'utf8'));
        if (meta?.repoPath) repoPaths.add(meta.repoPath);
      } catch { /* unreadable metadata: fall back to plain removal */ }
    }

    const runDirResolved = path.resolve(runDir);
    let removed = 0;
    for (const rp of repoPaths) {
      const listed = await git(['worktree', 'list', '--porcelain'], { cwd: rp });
      if (listed.exitCode !== 0) continue;
      for (const line of splitLines(listed.stdout)) {
        if (!line.startsWith('worktree ')) continue;
        const candidatePath = path.resolve(line.slice('worktree '.length).trim());
        if (!isInside(runDirResolved, candidatePath)) continue;
        const res = await git(['worktree', 'remove', '--force', candidatePath], { cwd: rp });
        if (res.exitCode === 0) removed += 1;
      }
      await git(['worktree', 'prune'], { cwd: rp });
    }
    await fs.rm(runDir, { recursive: true, force: true }).catch(() => {});
    return { removed };
  }

  return {
    kind: 'local',
    worktreesDir,
    create,
    open,
    destroyRun,
  };
}

function isInside(root, candidate) {
  const a = IS_WINDOWS ? root.toLowerCase() : root;
  const b = IS_WINDOWS ? candidate.toLowerCase() : candidate;
  return b === a || b.startsWith(a + path.sep);
}

function ensureTrailingNewline(text) {
  return text.endsWith('\n') ? text : `${text}\n`;
}

export default createLocalGitDriver;
