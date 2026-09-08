// kolm code — cloudflare sandbox worktree driver (spec sections 4 and 8).
//
// The driver is an HTTP client, so every test injects a fake fetch and asserts
// on the request it would have made. Nothing here opens a socket.

import test from 'node:test';
import assert from 'node:assert/strict';

import { createCloudflareSandboxDriver } from '../src/ladder/worktree/cloudflare-sandbox.js';
import { createDriver } from '../src/ladder/worktree/index.js';

const SANDBOX_URL = 'https://kolm-code-sandbox.example.workers.dev';
const SANDBOX_TOKEN = 'sandbox-token-value';
const BASE_SHA = '3333333333333333333333333333333333333333';

/** A fetch that records calls and answers from a queue of responses. */
function fakeFetch(responses = []) {
  const queue = [...responses];
  const calls = [];
  const impl = async (url, init = {}) => {
    const body = init.body === undefined ? undefined : JSON.parse(init.body);
    calls.push({ url, method: init.method, headers: init.headers ?? {}, body });
    const next = queue.length > 0 ? queue.shift() : { status: 200, body: {} };
    if (next instanceof Error) throw next;
    const text = typeof next.text === 'string' ? next.text : JSON.stringify(next.body ?? {});
    return {
      status: next.status ?? 200,
      ok: (next.status ?? 200) >= 200 && (next.status ?? 200) < 300,
      async text() {
        return text;
      },
    };
  };
  impl.calls = calls;
  return impl;
}

function armedDriver(responses) {
  const fetchImpl = fakeFetch(responses);
  const driver = createCloudflareSandboxDriver(
    { sandboxUrl: SANDBOX_URL, sandboxToken: SANDBOX_TOKEN },
    { fetchImpl },
  );
  return { driver, fetchImpl };
}

const CREATED = { status: 200, body: { id: 'cand1', path: '/workspace/run1/cand1', baseSha: BASE_SHA } };

async function armedWorktree(responses = []) {
  const { driver, fetchImpl } = armedDriver([CREATED, ...responses]);
  const wt = await driver.create({
    repoUrl: 'https://github.com/acme/widgets.git',
    baseRef: 'main',
    runId: 'run1',
    candidateId: 'cand1',
  });
  return { driver, wt, fetchImpl };
}

/* ---------------------------------------------------------------- arming */

test('an unarmed driver reports itself unarmed and refuses to be used', async () => {
  const driver = createCloudflareSandboxDriver({}, { fetchImpl: fakeFetch() });
  assert.equal(driver.kind, 'cloudflare');
  assert.equal(driver.armed, false);
  await assert.rejects(
    () => driver.create({ repoUrl: 'https://github.com/acme/widgets.git', runId: 'run1', candidateId: 'cand1' }),
    (err) => err.code === 'not-armed',
  );
  await assert.rejects(() => driver.destroyRun('run1'), (err) => err.code === 'not-armed');
});

test('a driver with a url but no token is still unarmed', async () => {
  const driver = createCloudflareSandboxDriver({ sandboxUrl: SANDBOX_URL }, { fetchImpl: fakeFetch() });
  assert.equal(driver.armed, false);
  await assert.rejects(
    () => driver.open({ runId: 'run1', candidateId: 'cand1' }),
    (err) => err.code === 'not-armed' && err.detail.hasUrl === true && err.detail.hasToken === false,
  );
});

test('an armed driver exposes its normalised base url', () => {
  const { driver } = armedDriver();
  assert.equal(driver.armed, true);
  assert.equal(driver.sandboxUrl, SANDBOX_URL);
  const trailing = createCloudflareSandboxDriver(
    { sandboxUrl: `${SANDBOX_URL}//`, sandboxToken: SANDBOX_TOKEN },
    { fetchImpl: fakeFetch() },
  );
  assert.equal(trailing.sandboxUrl, SANDBOX_URL);
  assert.equal(createDriver('cloudflare', { sandboxUrl: SANDBOX_URL, sandboxToken: SANDBOX_TOKEN }).armed, true);
});

/* ------------------------------------------------------------- creation */

test('creating a worktree posts the repository url ref and candidate id', async () => {
  const { driver, fetchImpl } = armedDriver([CREATED]);
  const wt = await driver.create({
    repoUrl: 'https://github.com/acme/widgets.git',
    baseRef: 'feature/x',
    runId: 'run1',
    candidateId: 'cand1',
    gitToken: 'installation-token',
  });

  assert.equal(fetchImpl.calls.length, 1);
  const call = fetchImpl.calls[0];
  assert.equal(call.url, `${SANDBOX_URL}/v1/runs/run1/worktrees`);
  assert.equal(call.method, 'POST');
  assert.equal(call.headers.authorization, `Bearer ${SANDBOX_TOKEN}`);
  assert.equal(call.headers['content-type'], 'application/json');
  assert.deepEqual(call.body, {
    repoUrl: 'https://github.com/acme/widgets.git',
    baseRef: 'feature/x',
    candidateId: 'cand1',
    gitToken: 'installation-token',
  });

  assert.equal(wt.kind, 'cloudflare');
  assert.equal(wt.id, 'cand1');
  assert.equal(wt.path, '/workspace/run1/cand1');
  assert.equal(wt.branch, 'kolm/code/run1/cand1');
  assert.equal(await wt.baseSha(), BASE_SHA);
});

test('creating without a repository url is a configuration error', async () => {
  const { driver, fetchImpl } = armedDriver();
  await assert.rejects(
    () => driver.create({ runId: 'run1', candidateId: 'cand1' }),
    (err) => err.code === 'bad-config',
  );
  assert.equal(fetchImpl.calls.length, 0);
});

test('a run identifier with unusual characters is encoded into the path', async () => {
  const { driver, fetchImpl } = armedDriver([CREATED]);
  await driver.create({
    repoUrl: 'https://github.com/acme/widgets.git',
    runId: 'run 1/x',
    candidateId: 'cand#1',
  });
  assert.equal(fetchImpl.calls[0].url, `${SANDBOX_URL}/v1/runs/run%201%2Fx/worktrees`);
});

test('open fetches the existing worktree rather than creating one', async () => {
  const { driver, fetchImpl } = armedDriver([CREATED]);
  const wt = await driver.open({ runId: 'run1', candidateId: 'cand1' });
  assert.equal(fetchImpl.calls[0].method, 'GET');
  assert.equal(fetchImpl.calls[0].url, `${SANDBOX_URL}/v1/runs/run1/worktrees/cand1`);
  assert.equal(await wt.baseSha(), BASE_SHA);
});

/* --------------------------------------------------------- worktree calls */

test('exec posts the command with its cwd timeout and environment', async () => {
  const { wt, fetchImpl } = await armedWorktree([
    { status: 200, body: { stdout: 'ok', stderr: '', exitCode: 0, durationMs: 12, timedOut: false, truncated: false } },
  ]);
  const result = await wt.exec('npm test --silent', { cwd: 'packages/web', timeoutMs: 4000, env: { CI: '1' } });
  const call = fetchImpl.calls[1];
  assert.equal(call.url, `${SANDBOX_URL}/v1/runs/run1/worktrees/cand1/exec`);
  assert.equal(call.method, 'POST');
  assert.deepEqual(call.body, {
    cmd: 'npm test --silent',
    cwd: 'packages/web',
    timeoutMs: 4000,
    env: { CI: '1' },
  });
  assert.deepEqual(result, {
    stdout: 'ok', stderr: '', exitCode: 0, durationMs: 12, timedOut: false, truncated: false,
  });
  await assert.rejects(() => wt.exec(''), (err) => err.code === 'bad-config');
});

test('exec falls back to the configured stage timeout', async () => {
  const fetchImpl = fakeFetch([CREATED, { status: 200, body: { exitCode: 0 } }]);
  const driver = createCloudflareSandboxDriver(
    { sandboxUrl: SANDBOX_URL, sandboxToken: SANDBOX_TOKEN, stageTimeoutMs: 45000 },
    { fetchImpl },
  );
  const wt = await driver.create({ repoUrl: 'https://github.com/a/b.git', runId: 'run1', candidateId: 'cand1' });
  await wt.exec('ls');
  assert.equal(fetchImpl.calls[1].body.timeoutMs, 45000);
});

test('file reads writes and deletes map onto the file route', async () => {
  const { wt, fetchImpl } = await armedWorktree([
    { status: 200, body: { path: '/workspace/run1/cand1/src/a.js', content: 'export const a = 1;\n' } },
    { status: 404, body: { error: 'not-found' } },
    { status: 200, body: { ok: true } },
    { status: 200, body: { ok: true } },
  ]);

  assert.equal(await wt.readFile('src/a.js'), 'export const a = 1;\n');
  assert.equal(fetchImpl.calls[1].url, `${SANDBOX_URL}/v1/runs/run1/worktrees/cand1/file?path=src%2Fa.js`);
  assert.equal(fetchImpl.calls[1].method, 'GET');

  assert.equal(await wt.readFile('src/missing.js'), null, 'a missing file reads as null, not an error');

  await wt.writeFile('src/b.js', 'export const b = 2;\n');
  assert.equal(fetchImpl.calls[3].method, 'PUT');
  assert.deepEqual(fetchImpl.calls[3].body, { path: 'src/b.js', content: 'export const b = 2;\n' });

  await wt.deleteFile('src/b.js');
  assert.equal(fetchImpl.calls[4].method, 'DELETE');
  assert.equal(fetchImpl.calls[4].url, `${SANDBOX_URL}/v1/runs/run1/worktrees/cand1/file?path=src%2Fb.js`);
});

test('listing and searching pass their glob and limit as query parameters', async () => {
  const { wt, fetchImpl } = await armedWorktree([
    { status: 200, body: { files: ['src/a.js'] } },
    { status: 200, body: { hits: [{ path: 'src/a.js', line: 1, text: 'const needle = 1;' }] } },
    { status: 200, body: {} },
  ]);

  assert.deepEqual(await wt.listFiles({ glob: 'src/**', limit: 10 }), ['src/a.js']);
  assert.equal(
    fetchImpl.calls[1].url,
    `${SANDBOX_URL}/v1/runs/run1/worktrees/cand1/files?glob=src%2F**&limit=10`,
  );

  assert.deepEqual(await wt.search('needle', { glob: '*.js' }), [
    { path: 'src/a.js', line: 1, text: 'const needle = 1;' },
  ]);
  assert.equal(
    fetchImpl.calls[2].url,
    `${SANDBOX_URL}/v1/runs/run1/worktrees/cand1/search?pattern=needle&glob=*.js&limit=200`,
  );

  assert.deepEqual(await wt.listFiles(), [], 'a response without files reads as an empty list');
});

test('diff and changed files read the review routes', async () => {
  const patch = 'diff --git a/src/a.js b/src/a.js\n@@ -1 +1 @@\n-a\n+b\n';
  const { wt, fetchImpl } = await armedWorktree([
    { status: 200, body: { diff: patch, truncated: false } },
    { status: 200, body: { files: [{ path: 'src/a.js', status: 'M', added: 1, deleted: 1 }] } },
  ]);
  assert.equal(await wt.diff(), patch);
  assert.equal(fetchImpl.calls[1].url, `${SANDBOX_URL}/v1/runs/run1/worktrees/cand1/diff`);
  assert.deepEqual(await wt.changedFiles(), [{ path: 'src/a.js', status: 'M', added: 1, deleted: 1 }]);
  assert.equal(fetchImpl.calls[2].url, `${SANDBOX_URL}/v1/runs/run1/worktrees/cand1/changed`);
});

test('applying a patch reports success and maps a rejected patch to ok false', async () => {
  const { wt, fetchImpl } = await armedWorktree([
    { status: 200, body: { ok: true } },
    { status: 200, body: { ok: false, error: 'patch does not apply' } },
    { status: 422, body: { error: 'patch failed' } },
  ]);
  assert.deepEqual(await wt.applyPatch('diff --git a/x b/x\n'), { ok: true, error: undefined });
  assert.equal(fetchImpl.calls[1].method, 'POST');
  assert.deepEqual(fetchImpl.calls[1].body, { diff: 'diff --git a/x b/x\n' });

  const rejected = await wt.applyPatch('diff --git a/x b/x\n');
  assert.equal(rejected.ok, false);
  assert.equal(rejected.error, 'patch does not apply');

  const failed = await wt.applyPatch('diff --git a/x b/x\n');
  assert.equal(failed.ok, false, 'a 422 from the sandbox is a failed patch, not a transport error');
  assert.equal(failed.error, 'patch failed');
});

test('committing returns the sha and the candidate branch', async () => {
  const sha = '4444444444444444444444444444444444444444';
  const { wt, fetchImpl } = await armedWorktree([
    { status: 200, body: { sha, branch: 'kolm/code/run1/cand1' } },
  ]);
  assert.deepEqual(await wt.commit('candidate work'), { sha, branch: 'kolm/code/run1/cand1' });
  assert.equal(fetchImpl.calls[1].url, `${SANDBOX_URL}/v1/runs/run1/worktrees/cand1/commit`);
  assert.deepEqual(fetchImpl.calls[1].body, { message: 'candidate work' });
});

test('destroying a worktree and a run use the delete routes', async () => {
  const { driver, wt, fetchImpl } = await armedWorktree([
    { status: 200, body: { ok: true } },
    { status: 200, body: { ok: true } },
  ]);
  assert.deepEqual(await wt.destroy(), { removed: true });
  assert.equal(fetchImpl.calls[1].method, 'DELETE');
  assert.equal(fetchImpl.calls[1].url, `${SANDBOX_URL}/v1/runs/run1/worktrees/cand1`);

  assert.deepEqual(await driver.destroyRun('run1'), { removed: true });
  assert.equal(fetchImpl.calls[2].method, 'DELETE');
  assert.equal(fetchImpl.calls[2].url, `${SANDBOX_URL}/v1/runs/run1`);
});

/* -------------------------------------------------------- error mapping */

test('a non success status becomes a sandbox http error carrying the body', async () => {
  const { wt } = await armedWorktree([{ status: 500, body: { error: 'internal', message: 'boom' } }]);
  await assert.rejects(
    () => wt.diff(),
    (err) => {
      assert.equal(err.code, 'sandbox-http');
      assert.equal(err.detail.status, 500);
      assert.deepEqual(err.detail.body, { error: 'internal', message: 'boom' });
      assert.match(err.message, /GET \/v1\/runs\/run1\/worktrees\/cand1\/diff -> 500/);
      return true;
    },
  );
});

test('an unauthorized sandbox response is surfaced with its status', async () => {
  const { driver } = armedDriver([{ status: 401, body: { error: 'unauthorized' } }]);
  await assert.rejects(
    () => driver.create({ repoUrl: 'https://github.com/a/b.git', runId: 'run1', candidateId: 'cand1' }),
    (err) => err.code === 'sandbox-http' && err.detail.status === 401,
  );
});

test('a transport failure is reported as a sandbox http error with status zero', async () => {
  const { driver } = armedDriver([new Error('connection refused')]);
  await assert.rejects(
    () => driver.destroyRun('run1'),
    (err) => err.code === 'sandbox-http' && err.detail.status === 0 && /connection refused/.test(err.message),
  );
});

test('a driver whose fetch implementation is not callable refuses to make requests', async () => {
  const driver = createCloudflareSandboxDriver(
    { sandboxUrl: SANDBOX_URL, sandboxToken: SANDBOX_TOKEN },
    { fetchImpl: 'not a function' },
  );
  await assert.rejects(() => driver.destroyRun('run1'), (err) => err.code === 'not-armed');
});
