// kolm code sandbox — pure routing layer (spec section 8).
//
// Everything here is a function of `getSandbox(id)`; there is no Cloudflare
// import in this file, so the whole surface is unit-testable against a fake
// sandbox recorder (`test/api.test.mjs`).
//
// Layout inside the container, one container per run:
//   /workspace/<runId>/base                 shallow clone (created once)
//   /workspace/<runId>/<candidateId>        detached worktree per candidate
//   /workspace/<runId>/<candidateId>.base   the base sha for that worktree
//
// The git credential (a GitHub App installation token) is only ever passed as
// an `-c http.extraheader=...` argument on the clone/fetch command. It is never
// written to a file, a git config, or a response body.

import { timingSafeEqual } from 'node:crypto';

export const WORKSPACE_ROOT = '/workspace';
export const MAX_OUTPUT_BYTES = 1 << 20;
export const DEFAULT_TIMEOUT_MS = 300000;
export const MAX_TIMEOUT_MS = 900000;
export const CLONE_DEPTH = 50;
export const GIT_IDENTITY = "-c user.name='kolm code' -c user.email='code@kolm.ai'";

const SEGMENT_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/;
const REPO_URL_RE = /^https:\/\/[^\s'"\\\x00-\x1f]{3,2000}$/;
const GIT_TOKEN_RE = /^[A-Za-z0-9_.-]{1,512}$/;

class ApiError extends Error {
  constructor(status, code, message) {
    super(message ?? code);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
  }
}

/** POSIX single-quote a shell word. */
export function shellQuote(value) {
  return `'${String(value).replace(/'/g, "'\\''")}'`;
}

function json(status, body) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json; charset=utf-8' },
  });
}

function assertSegment(value, field) {
  if (typeof value !== 'string' || !SEGMENT_RE.test(value)) {
    throw new ApiError(400, 'invalid-id', `${field} must match ${SEGMENT_RE}`);
  }
  return value;
}

/** Resolve `.`/`..` in a POSIX path without touching the filesystem. */
export function normalizePosix(input) {
  const absolute = input.startsWith('/');
  const out = [];
  for (const segment of input.split('/')) {
    if (segment === '' || segment === '.') continue;
    if (segment === '..') {
      if (out.length === 0) return null;
      out.pop();
      continue;
    }
    out.push(segment);
  }
  return (absolute ? '/' : '') + out.join('/');
}

/**
 * Resolve a caller-supplied relative path inside `root`. Absolute paths, `..`
 * segments, backslashes and control characters are all rejected; the resolved
 * result must still be inside `root`.
 */
export function resolveInside(root, rel, field = 'path') {
  if (typeof rel !== 'string' || rel.length === 0) {
    throw new ApiError(400, 'invalid-path', `${field} must be a non-empty relative path`);
  }
  if (rel.length > 4096) throw new ApiError(400, 'invalid-path', `${field} is too long`);
  // eslint-disable-next-line no-control-regex
  if (/[\x00-\x1f\x7f\\]/.test(rel)) {
    throw new ApiError(400, 'invalid-path', `${field} contains a forbidden character`);
  }
  if (rel.startsWith('/') || /^[A-Za-z]:/.test(rel)) {
    throw new ApiError(400, 'invalid-path', `${field} must not be absolute`);
  }
  if (rel.split('/').some((segment) => segment === '..')) {
    throw new ApiError(400, 'invalid-path', `${field} must not contain ".."`);
  }
  const resolved = normalizePosix(`${root}/${rel}`);
  if (!resolved || (resolved !== root && !resolved.startsWith(`${root}/`))) {
    throw new ApiError(400, 'invalid-path', `${field} escapes the worktree`);
  }
  return resolved;
}

function clip(text) {
  const value = typeof text === 'string' ? text : '';
  if (value.length <= MAX_OUTPUT_BYTES) return { text: value, truncated: false };
  return { text: value.slice(0, MAX_OUTPUT_BYTES), truncated: true };
}

function clampTimeout(value) {
  const n = Number(value);
  if (!Number.isFinite(n) || n <= 0) return DEFAULT_TIMEOUT_MS;
  return Math.min(Math.floor(n), MAX_TIMEOUT_MS);
}

/**
 * @param {object} deps
 * @param {(id: string) => object} deps.getSandbox
 * @param {string|undefined} deps.token   value of KOLM_CODE_SANDBOX_TOKEN
 * @param {() => number} [deps.now]
 */
export function createApi({ getSandbox, token, now = () => Date.now() }) {
  if (typeof getSandbox !== 'function') {
    throw new TypeError('createApi requires getSandbox(id)');
  }

  function authorized(request) {
    const header = request.headers.get('authorization') ?? '';
    const match = /^Bearer[ ]+(.*)$/i.exec(header);
    if (!match) return false;
    const encoder = new TextEncoder();
    const given = encoder.encode(match[1]);
    const expected = encoder.encode(token);
    if (given.byteLength !== expected.byteLength) return false;
    return timingSafeEqual(given, expected);
  }

  /** Run a shell script inside the container under a hard `timeout`. */
  async function sh(sandbox, script, { cwd, timeoutMs = DEFAULT_TIMEOUT_MS } = {}) {
    const seconds = Math.max(1, Math.ceil(clampTimeout(timeoutMs) / 1000));
    const started = now();
    const command = `timeout --kill-after=5s ${seconds}s sh -c ${shellQuote(script)}`;
    let result;
    try {
      result = await sandbox.exec(command, { cwd, timeout: clampTimeout(timeoutMs) + 5000 });
    } catch (err) {
      throw new ApiError(502, 'sandbox-exec-failed', String(err?.message ?? err));
    }
    const stdout = clip(result?.stdout);
    const stderr = clip(result?.stderr);
    const exitCode = typeof result?.exitCode === 'number' ? result.exitCode : -1;
    return {
      stdout: stdout.text,
      stderr: stderr.text,
      exitCode,
      durationMs: typeof result?.duration === 'number' ? result.duration : now() - started,
      timedOut: exitCode === 124 || exitCode === 137,
      truncated: stdout.truncated || stderr.truncated,
    };
  }

  async function shOrThrow(sandbox, script, options, code) {
    const result = await sh(sandbox, script, options);
    if (result.exitCode !== 0) {
      throw new ApiError(422, code, (result.stderr || result.stdout).trim() || `exited ${result.exitCode}`);
    }
    return result;
  }

  /** Refuse a path whose realpath leaves the worktree (symlink escape). */
  async function assertRealpathInside(sandbox, root, absolute) {
    const result = await sh(sandbox, `realpath -m -- ${shellQuote(absolute)}`, { timeoutMs: 15000 });
    if (result.exitCode !== 0) {
      throw new ApiError(400, 'invalid-path', 'cannot resolve path inside the worktree');
    }
    const real = normalizePosix(result.stdout.trim());
    if (!real || (real !== root && !real.startsWith(`${root}/`))) {
      throw new ApiError(400, 'invalid-path', 'path escapes the worktree');
    }
    return real;
  }

  const runDir = (runId) => `${WORKSPACE_ROOT}/${runId}`;
  const baseDir = (runId) => `${runDir(runId)}/base`;
  const worktreeDir = (runId, id) => `${runDir(runId)}/${id}`;
  const baseShaFile = (runId, id) => `${runDir(runId)}/${id}.base`;

  async function readBaseSha(sandbox, runId, id) {
    try {
      const file = await sandbox.readFile(baseShaFile(runId, id), { encoding: 'utf-8' });
      const content = file?.content;
      if (typeof content === 'string' && content.trim()) return content.trim();
    } catch { /* fall through */ }
    throw new ApiError(404, 'worktree-not-found', `no worktree ${id} in run ${runId}`);
  }

  async function readBody(request) {
    let text = '';
    try {
      text = await request.text();
    } catch {
      throw new ApiError(400, 'invalid-body', 'request body could not be read');
    }
    if (!text) return {};
    try {
      const parsed = JSON.parse(text);
      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
        throw new Error('body must be a JSON object');
      }
      return parsed;
    } catch (err) {
      throw new ApiError(400, 'invalid-body', String(err?.message ?? err));
    }
  }

  /* ------------------------------------------------------------ handlers */

  async function createWorktree(sandbox, runId, body) {
    const candidateId = assertSegment(body.candidateId, 'candidateId');
    const repoUrl = body.repoUrl;
    if (typeof repoUrl !== 'string' || !REPO_URL_RE.test(repoUrl)) {
      throw new ApiError(400, 'invalid-repo-url', 'repoUrl must be an https URL');
    }
    const baseRef = typeof body.baseRef === 'string' && body.baseRef ? body.baseRef : 'HEAD';
    if (!/^[A-Za-z0-9._/^~{}@-]{1,255}$/.test(baseRef)) {
      throw new ApiError(400, 'invalid-ref', 'baseRef contains forbidden characters');
    }
    const gitToken = body.gitToken;
    if (gitToken !== undefined && (typeof gitToken !== 'string' || !GIT_TOKEN_RE.test(gitToken))) {
      throw new ApiError(400, 'invalid-token', 'gitToken has an unexpected shape');
    }
    // Basic auth with the literal user `x-access-token` is the form GitHub
    // accepts for App installation tokens over https.
    const authArg = gitToken
      ? `-c ${shellQuote(`http.extraheader=AUTHORIZATION: Basic ${btoa(`x-access-token:${gitToken}`)}`)} `
      : '';

    const base = baseDir(runId);
    const existing = await sh(sandbox, `test -d ${shellQuote(`${base}/.git`)}`, { timeoutMs: 15000 });
    let cloned = false;
    if (existing.exitCode !== 0) {
      await shOrThrow(sandbox, `mkdir -p ${shellQuote(runDir(runId))}`, { timeoutMs: 15000 }, 'mkdir-failed');
      await shOrThrow(
        sandbox,
        `GIT_TERMINAL_PROMPT=0 git ${authArg}-c credential.helper= clone --depth ${CLONE_DEPTH} --no-single-branch ${shellQuote(repoUrl)} ${shellQuote(base)}`,
        { timeoutMs: 600000 },
        'clone-failed',
      );
      cloned = true;
    }

    let resolved = await sh(sandbox, `git rev-parse ${shellQuote(`${baseRef}^{commit}`)}`, {
      cwd: base,
      timeoutMs: 60000,
    });
    if (resolved.exitCode !== 0) {
      await shOrThrow(
        sandbox,
        `GIT_TERMINAL_PROMPT=0 git ${authArg}-c credential.helper= fetch --depth ${CLONE_DEPTH} origin ${shellQuote(baseRef)}`,
        { cwd: base, timeoutMs: 300000 },
        'fetch-failed',
      );
      resolved = await sh(sandbox, 'git rev-parse FETCH_HEAD^{commit}', { cwd: base, timeoutMs: 60000 });
    }
    if (resolved.exitCode !== 0) {
      throw new ApiError(422, 'bad-ref', `cannot resolve ${baseRef}`);
    }
    const baseSha = resolved.stdout.trim();

    const wtPath = worktreeDir(runId, candidateId);
    const present = await sh(sandbox, `test -e ${shellQuote(`${wtPath}/.git`)}`, { timeoutMs: 15000 });
    if (present.exitCode !== 0) {
      await shOrThrow(
        sandbox,
        `git worktree add --detach ${shellQuote(wtPath)} ${shellQuote(baseSha)}`,
        { cwd: base, timeoutMs: 300000 },
        'worktree-failed',
      );
    }
    await sandbox.writeFile(baseShaFile(runId, candidateId), `${baseSha}\n`, { encoding: 'utf-8' });
    return json(200, { id: candidateId, path: wtPath, baseSha, cloned, reused: present.exitCode === 0 });
  }

  async function execInWorktree(sandbox, runId, id, body) {
    const root = worktreeDir(runId, id);
    await readBaseSha(sandbox, runId, id);
    const cmd = body.cmd;
    if (typeof cmd !== 'string' || cmd.trim() === '') {
      throw new ApiError(400, 'invalid-command', 'cmd must be a non-empty string');
    }
    if (cmd.length > 65536) throw new ApiError(400, 'invalid-command', 'cmd is too long');
    let cwd = root;
    if (body.cwd !== undefined && body.cwd !== null && body.cwd !== '') {
      cwd = resolveInside(root, body.cwd, 'cwd');
      await assertRealpathInside(sandbox, root, cwd);
    }
    const env = {};
    if (body.env && typeof body.env === 'object' && !Array.isArray(body.env)) {
      for (const [k, v] of Object.entries(body.env)) {
        if (/^[A-Za-z_][A-Za-z0-9_]*$/.test(k) && typeof v === 'string') env[k] = v;
      }
    }
    const timeoutMs = clampTimeout(body.timeoutMs);
    const seconds = Math.max(1, Math.ceil(timeoutMs / 1000));
    const started = now();
    const command = `timeout --kill-after=5s ${seconds}s sh -c ${shellQuote(cmd)}`;
    let result;
    try {
      result = await sandbox.exec(command, { cwd, timeout: timeoutMs + 5000, env });
    } catch (err) {
      throw new ApiError(502, 'sandbox-exec-failed', String(err?.message ?? err));
    }
    const cap = Number.isFinite(Number(body.maxOutputBytes))
      ? Math.min(Number(body.maxOutputBytes), MAX_OUTPUT_BYTES)
      : MAX_OUTPUT_BYTES;
    const stdout = String(result?.stdout ?? '');
    const stderr = String(result?.stderr ?? '');
    const exitCode = typeof result?.exitCode === 'number' ? result.exitCode : -1;
    return json(200, {
      stdout: stdout.slice(0, cap),
      stderr: stderr.slice(0, cap),
      exitCode,
      durationMs: typeof result?.duration === 'number' ? result.duration : now() - started,
      timedOut: exitCode === 124 || exitCode === 137,
      truncated: stdout.length > cap || stderr.length > cap,
    });
  }

  async function getFile(sandbox, runId, id, url) {
    const root = worktreeDir(runId, id);
    const absolute = resolveInside(root, url.searchParams.get('path') ?? '', 'path');
    await assertRealpathInside(sandbox, root, absolute);
    let file;
    try {
      file = await sandbox.readFile(absolute, { encoding: 'utf-8' });
    } catch {
      return json(404, { error: 'not-found' });
    }
    if (!file || file.success === false || typeof file.content !== 'string') {
      return json(404, { error: 'not-found' });
    }
    return json(200, { path: absolute, content: file.content });
  }

  async function putFile(sandbox, runId, id, body) {
    const root = worktreeDir(runId, id);
    const absolute = resolveInside(root, body.path, 'path');
    await assertRealpathInside(sandbox, root, absolute);
    const content = typeof body.content === 'string' ? body.content : '';
    const parent = absolute.slice(0, absolute.lastIndexOf('/')) || root;
    await sandbox.mkdir(parent, { recursive: true });
    await sandbox.writeFile(absolute, content, { encoding: 'utf-8' });
    return json(200, { ok: true, path: absolute, bytes: content.length });
  }

  async function deleteFile(sandbox, runId, id, url) {
    const root = worktreeDir(runId, id);
    const absolute = resolveInside(root, url.searchParams.get('path') ?? '', 'path');
    await assertRealpathInside(sandbox, root, absolute);
    try {
      await sandbox.deleteFile(absolute);
    } catch { /* deleting a missing file is not an error */ }
    return json(200, { ok: true, path: absolute });
  }

  async function listFiles(sandbox, runId, id, url) {
    const root = worktreeDir(runId, id);
    await readBaseSha(sandbox, runId, id);
    const glob = url.searchParams.get('glob');
    const limit = Math.min(Number(url.searchParams.get('limit')) || 5000, 20000);
    const pathspec = glob ? ` -- ${shellQuote(glob)}` : '';
    const listed = await sh(sandbox, `git ls-files --cached --others --exclude-standard -z${pathspec}`, {
      cwd: root,
      timeoutMs: 120000,
    });
    if (listed.exitCode !== 0) throw new ApiError(422, 'git-failed', listed.stderr.trim());
    const deleted = await sh(sandbox, 'git ls-files --deleted -z', { cwd: root, timeoutMs: 120000 });
    const gone = new Set(deleted.stdout.split('\0').filter(Boolean));
    const files = [];
    const seen = new Set();
    for (const entry of listed.stdout.split('\0')) {
      if (!entry || seen.has(entry) || gone.has(entry)) continue;
      seen.add(entry);
      files.push(entry);
    }
    files.sort();
    return json(200, { files: files.slice(0, limit) });
  }

  async function search(sandbox, runId, id, url) {
    const root = worktreeDir(runId, id);
    await readBaseSha(sandbox, runId, id);
    const pattern = url.searchParams.get('pattern');
    if (!pattern) throw new ApiError(400, 'invalid-pattern', 'pattern is required');
    const glob = url.searchParams.get('glob');
    const limit = Math.min(Number(url.searchParams.get('limit')) || 200, 5000);
    const pathspec = glob ? ` -- ${shellQuote(glob)}` : '';
    const result = await sh(
      sandbox,
      `git -c core.quotepath=false grep -n -I --untracked -e ${shellQuote(pattern)}${pathspec}`,
      { cwd: root, timeoutMs: 120000 },
    );
    if (result.exitCode === 1) return json(200, { hits: [] });
    if (result.exitCode !== 0) throw new ApiError(422, 'git-failed', result.stderr.trim());
    const hits = [];
    for (const line of result.stdout.split('\n')) {
      if (!line || hits.length >= limit) continue;
      const first = line.indexOf(':');
      const second = line.indexOf(':', first + 1);
      if (first < 0 || second < 0) continue;
      const lineNo = Number(line.slice(first + 1, second));
      if (!Number.isFinite(lineNo)) continue;
      hits.push({ path: line.slice(0, first), line: lineNo, text: line.slice(second + 1) });
    }
    return json(200, { hits });
  }

  /** Tracked diff vs the base sha plus one `--no-index` diff per untracked file. */
  function diffScript(baseSha) {
    return [
      `git -c core.quotepath=false diff --no-ext-diff --no-color --no-renames ${shellQuote(baseSha)} --`,
      "git ls-files --others --exclude-standard -z | while IFS= read -r -d '' f; do"
        + ' git -c core.quotepath=false diff --no-ext-diff --no-color --no-index -- /dev/null "$f" || true;'
        + ' done',
    ].join('\n');
  }

  async function diff(sandbox, runId, id) {
    const root = worktreeDir(runId, id);
    const baseSha = await readBaseSha(sandbox, runId, id);
    const result = await sh(sandbox, diffScript(baseSha), { cwd: root, timeoutMs: 300000 });
    if (result.exitCode !== 0 && result.exitCode !== 1) {
      throw new ApiError(422, 'git-failed', result.stderr.trim());
    }
    return json(200, { diff: result.stdout, truncated: result.truncated });
  }

  function changedScript(baseSha) {
    return [
      `git -c core.quotepath=false diff --no-ext-diff --no-renames --numstat -z ${shellQuote(baseSha)} --`,
      'printf "\\036"',
      `git -c core.quotepath=false diff --no-ext-diff --no-renames --name-status -z ${shellQuote(baseSha)} --`,
      'printf "\\036"',
      "git ls-files --others --exclude-standard -z | while IFS= read -r -d '' f; do"
        + ' git diff --no-ext-diff --numstat --no-index -- /dev/null "$f" || true;'
        + ' done',
    ].join('\n');
  }

  async function changed(sandbox, runId, id) {
    const root = worktreeDir(runId, id);
    const baseSha = await readBaseSha(sandbox, runId, id);
    const result = await sh(sandbox, changedScript(baseSha), { cwd: root, timeoutMs: 300000 });
    if (result.exitCode !== 0 && result.exitCode !== 1) {
      throw new ApiError(422, 'git-failed', result.stderr.trim());
    }
    const [numstatRaw = '', nameStatusRaw = '', untrackedRaw = ''] = result.stdout.split('\x1e');
    const statusByPath = new Map();
    const nameFields = nameStatusRaw.split('\0').filter((f) => f !== '');
    for (let i = 0; i + 1 < nameFields.length; i += 2) {
      statusByPath.set(nameFields[i + 1], nameFields[i].charAt(0));
    }
    const files = [];
    for (const record of numstatRaw.split('\0')) {
      if (!record) continue;
      const tab1 = record.indexOf('\t');
      const tab2 = record.indexOf('\t', tab1 + 1);
      if (tab1 < 0 || tab2 < 0) continue;
      const added = record.slice(0, tab1);
      const deleted = record.slice(tab1 + 1, tab2);
      const p = record.slice(tab2 + 1);
      files.push({
        path: p,
        status: statusByPath.get(p) ?? 'M',
        added: added === '-' ? 0 : Number(added) || 0,
        deleted: deleted === '-' ? 0 : Number(deleted) || 0,
      });
    }
    for (const line of untrackedRaw.split('\n')) {
      if (!line.trim()) continue;
      const parts = line.split('\t');
      if (parts.length < 3) continue;
      files.push({
        path: parts[2],
        status: 'A',
        added: parts[0] === '-' ? 0 : Number(parts[0]) || 0,
        deleted: 0,
      });
    }
    files.sort((a, b) => (a.path < b.path ? -1 : a.path > b.path ? 1 : 0));
    return json(200, { files });
  }

  async function patch(sandbox, runId, id, body) {
    const root = worktreeDir(runId, id);
    await readBaseSha(sandbox, runId, id);
    const diffText = body.diff;
    if (typeof diffText !== 'string' || diffText.trim() === '') {
      return json(200, { ok: false, error: 'empty patch' });
    }
    const file = `${runDir(runId)}/.${id}.patch`;
    await sandbox.writeFile(file, diffText.endsWith('\n') ? diffText : `${diffText}\n`, {
      encoding: 'utf-8',
    });
    const result = await sh(sandbox, `git apply --3way --whitespace=nowarn ${shellQuote(file)}`, {
      cwd: root,
      timeoutMs: 120000,
    });
    try {
      await sandbox.deleteFile(file);
    } catch { /* best effort */ }
    if (result.exitCode === 0) return json(200, { ok: true });
    return json(200, { ok: false, error: (result.stderr || result.stdout).trim() });
  }

  async function commit(sandbox, runId, id, body) {
    const root = worktreeDir(runId, id);
    await readBaseSha(sandbox, runId, id);
    const message = typeof body.message === 'string' ? body.message : '';
    const branch = `kolm/code/${runId}/${id}`;
    // `git add -A` is confined to this throwaway candidate worktree.
    await shOrThrow(sandbox, 'git add -A', { cwd: root, timeoutMs: 120000 }, 'git-failed');
    await shOrThrow(
      sandbox,
      `git ${GIT_IDENTITY} commit -q -m ${shellQuote(message)} --allow-empty-message`,
      { cwd: root, timeoutMs: 120000 },
      'commit-failed',
    );
    const sha = await shOrThrow(sandbox, 'git rev-parse HEAD', { cwd: root, timeoutMs: 60000 }, 'git-failed');
    await shOrThrow(
      sandbox,
      `git branch -f ${shellQuote(branch)} ${shellQuote(sha.stdout.trim())}`,
      { cwd: root, timeoutMs: 60000 },
      'git-failed',
    );
    return json(200, { sha: sha.stdout.trim(), branch });
  }

  async function destroyWorktree(sandbox, runId, id) {
    const wtPath = worktreeDir(runId, id);
    await sh(
      sandbox,
      `git -C ${shellQuote(baseDir(runId))} worktree remove --force ${shellQuote(wtPath)} || rm -rf ${shellQuote(wtPath)}`,
      { timeoutMs: 120000 },
    );
    await sh(sandbox, `git -C ${shellQuote(baseDir(runId))} worktree prune`, { timeoutMs: 60000 });
    try {
      await sandbox.deleteFile(baseShaFile(runId, id));
    } catch { /* best effort */ }
    return json(200, { ok: true });
  }

  async function destroyRun(sandbox, runId) {
    await sh(sandbox, `rm -rf ${shellQuote(runDir(runId))}`, { timeoutMs: 120000 });
    if (typeof sandbox.destroy === 'function') {
      try {
        await sandbox.destroy();
      } catch { /* container teardown is best effort */ }
    }
    return json(200, { ok: true });
  }

  async function getWorktree(sandbox, runId, id) {
    const baseSha = await readBaseSha(sandbox, runId, id);
    return json(200, { id, path: worktreeDir(runId, id), baseSha });
  }

  /* ------------------------------------------------------------- routing */

  async function route(request, url) {
    const parts = url.pathname.split('/').filter(Boolean);
    if (parts[0] !== 'v1' || parts[1] !== 'runs' || parts.length < 3) {
      throw new ApiError(404, 'not-found', 'no such route');
    }
    const runId = assertSegment(decodeURIComponent(parts[2]), 'runId');
    const sandbox = getSandbox(runId);
    const method = request.method.toUpperCase();

    if (parts.length === 3) {
      if (method === 'DELETE') return destroyRun(sandbox, runId);
      throw new ApiError(405, 'method-not-allowed', `${method} not allowed here`);
    }
    if (parts[3] !== 'worktrees') throw new ApiError(404, 'not-found', 'no such route');

    if (parts.length === 4) {
      if (method === 'POST') return createWorktree(sandbox, runId, await readBody(request));
      throw new ApiError(405, 'method-not-allowed', `${method} not allowed here`);
    }

    const id = assertSegment(decodeURIComponent(parts[4]), 'worktreeId');

    if (parts.length === 5) {
      if (method === 'GET') return getWorktree(sandbox, runId, id);
      if (method === 'DELETE') return destroyWorktree(sandbox, runId, id);
      throw new ApiError(405, 'method-not-allowed', `${method} not allowed here`);
    }
    if (parts.length !== 6) throw new ApiError(404, 'not-found', 'no such route');

    switch (`${method} ${parts[5]}`) {
      case 'POST exec': return execInWorktree(sandbox, runId, id, await readBody(request));
      case 'GET file': return getFile(sandbox, runId, id, url);
      case 'PUT file': return putFile(sandbox, runId, id, await readBody(request));
      case 'DELETE file': return deleteFile(sandbox, runId, id, url);
      case 'GET files': return listFiles(sandbox, runId, id, url);
      case 'GET search': return search(sandbox, runId, id, url);
      case 'GET diff': return diff(sandbox, runId, id);
      case 'GET changed': return changed(sandbox, runId, id);
      case 'POST patch': return patch(sandbox, runId, id, await readBody(request));
      case 'POST commit': return commit(sandbox, runId, id, await readBody(request));
      default: throw new ApiError(404, 'not-found', 'no such route');
    }
  }

  async function handle(request) {
    try {
      if (typeof token !== 'string' || token.length === 0) {
        return json(500, { error: 'sandbox-not-configured', message: 'KOLM_CODE_SANDBOX_TOKEN is unset' });
      }
      if (!authorized(request)) return json(401, { error: 'unauthorized' });
      return await route(request, new URL(request.url));
    } catch (err) {
      if (err instanceof ApiError) {
        return json(err.status, { error: err.code, message: err.message });
      }
      return json(500, { error: 'internal', message: String(err?.message ?? err) });
    }
  }

  return { handle };
}

export default createApi;
