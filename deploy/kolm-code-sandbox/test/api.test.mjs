// Unit tests for the kolm code sandbox worker API. Every test drives
// `createApi` against a fake sandbox recorder, so no container, no network and
// no Cloudflare runtime is involved.

import test from 'node:test';
import assert from 'node:assert/strict';

import { createApi, normalizePosix, resolveInside, shellQuote } from '../src/api.mjs';
import { DEFAULT_ALLOWED_HOSTS, parseAllowedHosts } from '../src/hosts.mjs';

const TOKEN = 'test-token-0123456789';
const RS = '\x1e';
const BASE_SHA = '1111111111111111111111111111111111111111';

/** Strip the `timeout --kill-after=5s Ns sh -c '...'` wrapper the API adds. */
function innerScript(command) {
  const match = /^timeout --kill-after=5s \d+s sh -c '([\s\S]*)'$/.exec(command);
  if (!match) return command;
  return match[1].replace(/'\\''/g, "'");
}

function createFakeSandbox({ handlers = [], files = {} } = {}) {
  const sandbox = {
    calls: [],
    writes: [],
    deletes: [],
    mkdirs: [],
    destroyed: 0,
    files: new Map(Object.entries(files)),

    async exec(command, options = {}) {
      const script = innerScript(command);
      sandbox.calls.push({ command, script, cwd: options.cwd, timeout: options.timeout, env: options.env });
      for (const handler of handlers) {
        const matched = typeof handler.match === 'function'
          ? handler.match(script)
          : handler.match.test(script);
        if (matched) {
          const result = typeof handler.result === 'function' ? handler.result(script) : handler.result;
          return { stdout: '', stderr: '', exitCode: 0, duration: 3, ...result };
        }
      }
      const realpath = /^realpath -m -- '([\s\S]*)'$/.exec(script);
      if (realpath) return { stdout: `${realpath[1]}\n`, stderr: '', exitCode: 0, duration: 1 };
      return { stdout: '', stderr: '', exitCode: 0, duration: 1 };
    },

    async readFile(path) {
      if (!sandbox.files.has(path)) throw new Error(`no such file: ${path}`);
      return { success: true, path, content: sandbox.files.get(path) };
    },

    async writeFile(path, content) {
      sandbox.writes.push({ path, content });
      sandbox.files.set(path, content);
      return { success: true, path };
    },

    async mkdir(path, options) {
      sandbox.mkdirs.push({ path, options });
      return { success: true, path };
    },

    async deleteFile(path) {
      sandbox.deletes.push(path);
      sandbox.files.delete(path);
      return { success: true, path };
    },

    async destroy() {
      sandbox.destroyed += 1;
    },
  };
  return sandbox;
}

function makeApi(sandbox, { token = TOKEN } = {}) {
  return createApi({ getSandbox: () => sandbox, token, now: () => 1000 });
}

function request(method, path, { body, token = TOKEN, headers = {} } = {}) {
  const init = { method, headers: { ...headers } };
  if (token !== null) init.headers.authorization = `Bearer ${token}`;
  if (body !== undefined) {
    init.body = JSON.stringify(body);
    init.headers['content-type'] = 'application/json';
  }
  return new Request(`https://sandbox.example${path}`, init);
}

const withBaseSha = (runId, id) => ({ [`/workspace/${runId}/${id}.base`]: `${BASE_SHA}\n` });

/* ------------------------------------------------------------------ auth */

test('requests without an authorization header are rejected with 401', async () => {
  const api = makeApi(createFakeSandbox());
  const response = await api.handle(request('GET', '/v1/runs/r1/worktrees/c1', { token: null }));
  assert.equal(response.status, 401);
  assert.deepEqual(await response.json(), { error: 'unauthorized' });
});

test('requests carrying the wrong bearer token are rejected with 401', async () => {
  const sandbox = createFakeSandbox();
  const api = makeApi(sandbox);
  const sameLength = 'test-token-9876543210';
  assert.equal(sameLength.length, TOKEN.length);
  for (const wrong of [sameLength, 'short', `${TOKEN}extra`, '']) {
    const response = await api.handle(request('GET', '/v1/runs/r1/worktrees/c1', { token: wrong }));
    assert.equal(response.status, 401, `token ${JSON.stringify(wrong)} must not be accepted`);
  }
  assert.equal(sandbox.calls.length, 0, 'an unauthorized request must never reach the container');
});

test('an unconfigured worker reports 500 instead of accepting any token', async () => {
  const api = makeApi(createFakeSandbox(), { token: null });
  const response = await api.handle(request('GET', '/v1/runs/r1/worktrees/c1', { token: 'anything' }));
  assert.equal(response.status, 500);
  assert.equal((await response.json()).error, 'sandbox-not-configured');
});

/* --------------------------------------------------------- path handling */

test('path helpers resolve inside the worktree and reject escapes', () => {
  assert.equal(normalizePosix('/workspace/r1/c1/./src/../src/a.js'), '/workspace/r1/c1/src/a.js');
  assert.equal(normalizePosix('../etc'), null);
  assert.equal(resolveInside('/workspace/r1/c1', 'src/a.js'), '/workspace/r1/c1/src/a.js');
  for (const bad of ['../secret', 'a/../../b', '/etc/passwd', 'C:/windows', 'a\\b', '']) {
    assert.throws(
      () => resolveInside('/workspace/r1/c1', bad),
      (err) => err.code === 'invalid-path',
      `rejects ${JSON.stringify(bad)}`,
    );
  }
  assert.equal(shellQuote("it's"), "'it'\\''s'");
});

test('file routes reject traversal, absolute and percent encoded paths', async () => {
  const sandbox = createFakeSandbox({ files: withBaseSha('r1', 'c1') });
  const api = makeApi(sandbox);
  const cases = [
    ['GET', '/v1/runs/r1/worktrees/c1/file?path=../../etc/passwd'],
    ['GET', '/v1/runs/r1/worktrees/c1/file?path=%2e%2e%2f%2e%2e%2fetc%2fpasswd'],
    ['GET', '/v1/runs/r1/worktrees/c1/file?path=/etc/passwd'],
    ['DELETE', '/v1/runs/r1/worktrees/c1/file?path=..%2Fsibling'],
  ];
  for (const [method, path] of cases) {
    const response = await api.handle(request(method, path));
    assert.equal(response.status, 400, `${method} ${path}`);
    assert.equal((await response.json()).error, 'invalid-path');
  }
  const put = await api.handle(request('PUT', '/v1/runs/r1/worktrees/c1/file', {
    body: { path: '../outside.txt', content: 'x' },
  }));
  assert.equal(put.status, 400);
  assert.equal(sandbox.writes.length, 0, 'a rejected path must not be written');
});

test('a symlinked path whose realpath leaves the worktree is refused', async () => {
  const sandbox = createFakeSandbox({
    files: withBaseSha('r1', 'c1'),
    handlers: [{ match: /^realpath/, result: { stdout: '/etc/shadow\n' } }],
  });
  const api = makeApi(sandbox);
  const response = await api.handle(request('GET', '/v1/runs/r1/worktrees/c1/file?path=link'));
  assert.equal(response.status, 400);
  assert.equal((await response.json()).message, 'path escapes the worktree');
});

test('run and worktree identifiers are restricted to a safe alphabet', async () => {
  const api = makeApi(createFakeSandbox());
  for (const path of ['/v1/runs/..%2Fetc/worktrees/c1', '/v1/runs/r1/worktrees/a%20b', '/v1/runs/-r/worktrees/c1']) {
    const response = await api.handle(request('GET', path));
    assert.equal(response.status, 400, path);
    assert.equal((await response.json()).error, 'invalid-id');
  }
});

/* ------------------------------------------------------------ worktrees */

test('creating a worktree clones once and adds a detached worktree', async () => {
  const sandbox = createFakeSandbox({
    handlers: [
      { match: /^test -d '\/workspace\/r1\/base\/\.git'$/, result: { exitCode: 1 } },
      { match: /^test -e '\/workspace\/r1\/c1\/\.git'$/, result: { exitCode: 1 } },
      { match: /git rev-parse/, result: { stdout: `${BASE_SHA}\n` } },
    ],
  });
  const api = makeApi(sandbox);
  const response = await api.handle(request('POST', '/v1/runs/r1/worktrees', {
    body: { repoUrl: 'https://github.com/acme/widgets.git', baseRef: 'main', candidateId: 'c1' },
  }));
  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), {
    id: 'c1',
    path: '/workspace/r1/c1',
    baseSha: BASE_SHA,
    cloned: true,
    reused: false,
  });
  const scripts = sandbox.calls.map((c) => c.script);
  const clone = scripts.find((s) => s.includes('git clone') || s.includes('clone --depth'));
  assert.ok(clone, 'a clone command must be issued');
  assert.match(clone, /clone --depth 50 --no-single-branch 'https:\/\/github\.com\/acme\/widgets\.git' '\/workspace\/r1\/base'/);
  const add = sandbox.calls.find((c) => c.script.includes('worktree add'));
  assert.match(add.script, /^git worktree add --detach '\/workspace\/r1\/c1' '1{40}'$/);
  assert.equal(add.cwd, '/workspace/r1/base');
  assert.deepEqual(sandbox.writes, [{ path: '/workspace/r1/c1.base', content: `${BASE_SHA}\n` }]);
});

test('a second worktree in the same run reuses the existing clone', async () => {
  const sandbox = createFakeSandbox({
    handlers: [
      { match: /^test -d '\/workspace\/r1\/base\/\.git'$/, result: { exitCode: 0 } },
      { match: /^test -e /, result: { exitCode: 1 } },
      { match: /git rev-parse/, result: { stdout: `${BASE_SHA}\n` } },
    ],
  });
  const api = makeApi(sandbox);
  const response = await api.handle(request('POST', '/v1/runs/r1/worktrees', {
    body: { repoUrl: 'https://github.com/acme/widgets.git', candidateId: 'c2' },
  }));
  const body = await response.json();
  assert.equal(body.cloned, false);
  assert.equal(sandbox.calls.filter((c) => c.script.includes('clone')).length, 0);
});

test('an existing worktree directory is reported as reused rather than recreated', async () => {
  const sandbox = createFakeSandbox({
    handlers: [{ match: /git rev-parse/, result: { stdout: `${BASE_SHA}\n` } }],
  });
  const api = makeApi(sandbox);
  const body = await (await api.handle(request('POST', '/v1/runs/r1/worktrees', {
    body: { repoUrl: 'https://github.com/acme/widgets.git', candidateId: 'c1' },
  }))).json();
  assert.equal(body.reused, true);
  assert.equal(sandbox.calls.filter((c) => c.script.includes('worktree add')).length, 0);
});

test('the git credential only appears as an extraheader on the clone command', async () => {
  const sandbox = createFakeSandbox({
    handlers: [
      { match: /^test -d /, result: { exitCode: 1 } },
      { match: /^test -e /, result: { exitCode: 1 } },
      { match: /git rev-parse/, result: { stdout: `${BASE_SHA}\n` } },
    ],
  });
  const api = makeApi(sandbox);
  const secret = 'ghs-super-secret-token';
  await api.handle(request('POST', '/v1/runs/r1/worktrees', {
    body: {
      repoUrl: 'https://github.com/acme/widgets.git',
      candidateId: 'c1',
      gitToken: secret,
    },
  }));
  const carrying = sandbox.calls.filter((c) => c.command.includes(secret));
  assert.equal(carrying.length, 0, 'the raw credential must never be interpolated in cleartext');
  const withHeader = sandbox.calls.filter((c) => c.script.includes('http.extraheader'));
  assert.equal(withHeader.length, 1, 'only the clone carries the credential');
  const encoded = Buffer.from(`x-access-token:${secret}`, 'utf8').toString('base64');
  assert.ok(withHeader[0].script.includes(`AUTHORIZATION: Basic ${encoded}`));
  assert.ok(withHeader[0].script.includes('clone --depth'));
  for (const write of sandbox.writes) {
    assert.ok(!String(write.content).includes(secret), 'the credential must never be written to disk');
  }
});

test('a repository url that is not https is refused before any command runs', async () => {
  const sandbox = createFakeSandbox();
  const api = makeApi(sandbox);
  for (const repoUrl of ['git@github.com:acme/widgets.git', 'file:///etc', "https://x/'; rm -rf /"]) {
    const response = await api.handle(request('POST', '/v1/runs/r1/worktrees', {
      body: { repoUrl, candidateId: 'c1' },
    }));
    assert.equal(response.status, 400, repoUrl);
    assert.equal((await response.json()).error, 'invalid-repo-url');
  }
  assert.equal(sandbox.calls.length, 0);
});

test('reading a worktree that was never created returns 404', async () => {
  const api = makeApi(createFakeSandbox());
  const response = await api.handle(request('GET', '/v1/runs/r1/worktrees/missing'));
  assert.equal(response.status, 404);
  assert.equal((await response.json()).error, 'worktree-not-found');
});

test('an existing worktree reports its container path and base sha', async () => {
  const api = makeApi(createFakeSandbox({ files: withBaseSha('r1', 'c1') }));
  const body = await (await api.handle(request('GET', '/v1/runs/r1/worktrees/c1'))).json();
  assert.deepEqual(body, { id: 'c1', path: '/workspace/r1/c1', baseSha: BASE_SHA });
});

/* ----------------------------------------------------------------- exec */

test('exec passes the command through with a bounded timeout and worktree cwd', async () => {
  const sandbox = createFakeSandbox({
    files: withBaseSha('r1', 'c1'),
    handlers: [{ match: /npm test/, result: { stdout: 'ok\n', stderr: 'warn\n', exitCode: 3, duration: 42 } }],
  });
  const api = makeApi(sandbox);
  const response = await api.handle(request('POST', '/v1/runs/r1/worktrees/c1/exec', {
    body: { cmd: 'npm test --silent', timeoutMs: 12000, env: { CI: '1', 'bad key': 'x' } },
  }));
  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), {
    stdout: 'ok\n',
    stderr: 'warn\n',
    exitCode: 3,
    durationMs: 42,
    timedOut: false,
    truncated: false,
  });
  const call = sandbox.calls.at(-1);
  assert.equal(call.command, "timeout --kill-after=5s 12s sh -c 'npm test --silent'");
  assert.equal(call.cwd, '/workspace/r1/c1');
  assert.equal(call.timeout, 17000);
  assert.deepEqual(call.env, { CI: '1' });
});

test('exec honours a relative cwd inside the worktree and refuses one outside', async () => {
  const sandbox = createFakeSandbox({ files: withBaseSha('r1', 'c1') });
  const api = makeApi(sandbox);
  const ok = await api.handle(request('POST', '/v1/runs/r1/worktrees/c1/exec', {
    body: { cmd: 'ls', cwd: 'packages/web' },
  }));
  assert.equal(ok.status, 200);
  assert.equal(sandbox.calls.at(-1).cwd, '/workspace/r1/c1/packages/web');
  const bad = await api.handle(request('POST', '/v1/runs/r1/worktrees/c1/exec', {
    body: { cmd: 'ls', cwd: '../../etc' },
  }));
  assert.equal(bad.status, 400);
});

test('a command killed by the timeout is reported as timed out', async () => {
  const sandbox = createFakeSandbox({
    files: withBaseSha('r1', 'c1'),
    handlers: [{ match: /sleep/, result: { exitCode: 124 } }],
  });
  const api = makeApi(sandbox);
  const body = await (await api.handle(request('POST', '/v1/runs/r1/worktrees/c1/exec', {
    body: { cmd: 'sleep 900', timeoutMs: 1000 },
  }))).json();
  assert.equal(body.timedOut, true);
  assert.equal(body.exitCode, 124);
});

test('exec output is capped by the requested maxOutputBytes', async () => {
  const sandbox = createFakeSandbox({
    files: withBaseSha('r1', 'c1'),
    handlers: [{ match: /yes/, result: { stdout: 'x'.repeat(5000) } }],
  });
  const api = makeApi(sandbox);
  const body = await (await api.handle(request('POST', '/v1/runs/r1/worktrees/c1/exec', {
    body: { cmd: 'yes', maxOutputBytes: 100 },
  }))).json();
  assert.equal(body.stdout.length, 100);
  assert.equal(body.truncated, true);
});

test('an empty command is rejected rather than forwarded to the shell', async () => {
  const sandbox = createFakeSandbox({ files: withBaseSha('r1', 'c1') });
  const api = makeApi(sandbox);
  const response = await api.handle(request('POST', '/v1/runs/r1/worktrees/c1/exec', { body: { cmd: '  ' } }));
  assert.equal(response.status, 400);
  assert.equal((await response.json()).error, 'invalid-command');
});

/* ---------------------------------------------------------------- files */

test('file get put and delete address paths inside the worktree', async () => {
  const sandbox = createFakeSandbox({
    files: { ...withBaseSha('r1', 'c1'), '/workspace/r1/c1/src/a.js': 'export const a = 1;\n' },
  });
  const api = makeApi(sandbox);

  const got = await api.handle(request('GET', '/v1/runs/r1/worktrees/c1/file?path=src/a.js'));
  assert.equal(got.status, 200);
  assert.deepEqual(await got.json(), { path: '/workspace/r1/c1/src/a.js', content: 'export const a = 1;\n' });

  const missing = await api.handle(request('GET', '/v1/runs/r1/worktrees/c1/file?path=src/nope.js'));
  assert.equal(missing.status, 404);

  const put = await api.handle(request('PUT', '/v1/runs/r1/worktrees/c1/file', {
    body: { path: 'src/nested/b.js', content: 'b\n' },
  }));
  assert.equal(put.status, 200);
  assert.deepEqual(await put.json(), { ok: true, path: '/workspace/r1/c1/src/nested/b.js', bytes: 2 });
  assert.deepEqual(sandbox.mkdirs.at(-1), { path: '/workspace/r1/c1/src/nested', options: { recursive: true } });

  const removed = await api.handle(request('DELETE', '/v1/runs/r1/worktrees/c1/file?path=src/a.js'));
  assert.equal(removed.status, 200);
  assert.deepEqual(sandbox.deletes, ['/workspace/r1/c1/src/a.js']);

  const again = await api.handle(request('DELETE', '/v1/runs/r1/worktrees/c1/file?path=src/a.js'));
  assert.equal(again.status, 200, 'deleting a missing file is not an error');
});

test('listing files uses git and drops entries deleted from the working tree', async () => {
  const sandbox = createFakeSandbox({
    files: withBaseSha('r1', 'c1'),
    handlers: [
      { match: /ls-files --cached --others/, result: { stdout: 'src/a.js\0src/b.js\0gone.txt\0src/a.js\0' } },
      { match: /ls-files --deleted/, result: { stdout: 'gone.txt\0' } },
    ],
  });
  const api = makeApi(sandbox);
  const response = await api.handle(request('GET', '/v1/runs/r1/worktrees/c1/files?glob=src/**'));
  assert.deepEqual(await response.json(), { files: ['src/a.js', 'src/b.js'] });
  const listing = sandbox.calls.find((c) => c.script.includes('--cached --others'));
  assert.match(listing.script, /-- 'src\/\*\*'$/);
  assert.equal(listing.cwd, '/workspace/r1/c1');
});

test('search maps git grep output to path line and text triples', async () => {
  const sandbox = createFakeSandbox({
    files: withBaseSha('r1', 'c1'),
    handlers: [{ match: /grep -n -I/, result: { stdout: 'src/a.js:12:const needle = 1;\nsrc/b.js:3:needle()\n' } }],
  });
  const api = makeApi(sandbox);
  const response = await api.handle(request('GET', '/v1/runs/r1/worktrees/c1/search?pattern=needle&glob=*.js'));
  assert.deepEqual(await response.json(), {
    hits: [
      { path: 'src/a.js', line: 12, text: 'const needle = 1;' },
      { path: 'src/b.js', line: 3, text: 'needle()' },
    ],
  });
  assert.match(sandbox.calls.at(-1).script, /grep -n -I --untracked -e 'needle' -- '\*\.js'/);
});

test('search returns an empty hit list when git grep exits with code one', async () => {
  const sandbox = createFakeSandbox({
    files: withBaseSha('r1', 'c1'),
    handlers: [{ match: /grep -n -I/, result: { exitCode: 1 } }],
  });
  const api = makeApi(sandbox);
  const response = await api.handle(request('GET', '/v1/runs/r1/worktrees/c1/search?pattern=needle'));
  assert.deepEqual(await response.json(), { hits: [] });
});

/* -------------------------------------------------------- diff and patch */

test('diff combines the tracked diff with untracked file diffs', async () => {
  const patchText = 'diff --git a/src/a.js b/src/a.js\n@@ -1 +1 @@\n-a\n+b\n';
  const sandbox = createFakeSandbox({
    files: withBaseSha('r1', 'c1'),
    handlers: [{ match: /diff --no-ext-diff/, result: { stdout: patchText } }],
  });
  const api = makeApi(sandbox);
  const response = await api.handle(request('GET', '/v1/runs/r1/worktrees/c1/diff'));
  assert.deepEqual(await response.json(), { diff: patchText, truncated: false });
  const script = sandbox.calls.at(-1).script;
  assert.ok(script.includes(`diff --no-ext-diff --no-color --no-renames '${BASE_SHA}' --`));
  assert.ok(script.includes('--no-index -- /dev/null'), 'untracked files are diffed against /dev/null');
});

test('changed files merges numstat name status and untracked additions', async () => {
  const numstat = `3\t1\tsrc/a.js\0` + `0\t7\tsrc/gone.js\0`;
  const nameStatus = `M\0src/a.js\0` + `D\0src/gone.js\0`;
  const untracked = '5\t0\tsrc/new.js\n';
  const sandbox = createFakeSandbox({
    files: withBaseSha('r1', 'c1'),
    handlers: [{ match: /--numstat -z/, result: { stdout: `${numstat}${RS}${nameStatus}${RS}${untracked}` } }],
  });
  const api = makeApi(sandbox);
  const response = await api.handle(request('GET', '/v1/runs/r1/worktrees/c1/changed'));
  assert.deepEqual(await response.json(), {
    files: [
      { path: 'src/a.js', status: 'M', added: 3, deleted: 1 },
      { path: 'src/gone.js', status: 'D', added: 0, deleted: 7 },
      { path: 'src/new.js', status: 'A', added: 5, deleted: 0 },
    ],
  });
});

test('binary counts in numstat output are reported as zero line changes', async () => {
  const sandbox = createFakeSandbox({
    files: withBaseSha('r1', 'c1'),
    handlers: [{ match: /--numstat -z/, result: { stdout: `-\t-\tlogo.png\0${RS}M\0logo.png\0${RS}` } }],
  });
  const api = makeApi(sandbox);
  const body = await (await api.handle(request('GET', '/v1/runs/r1/worktrees/c1/changed'))).json();
  assert.deepEqual(body.files, [{ path: 'logo.png', status: 'M', added: 0, deleted: 0 }]);
});

test('applying a patch writes a temp file runs git apply and cleans up', async () => {
  const sandbox = createFakeSandbox({ files: withBaseSha('r1', 'c1') });
  const api = makeApi(sandbox);
  const response = await api.handle(request('POST', '/v1/runs/r1/worktrees/c1/patch', {
    body: { diff: 'diff --git a/x b/x\n@@ -0,0 +1 @@\n+x\n' },
  }));
  assert.deepEqual(await response.json(), { ok: true });
  assert.equal(sandbox.writes.at(-1).path, '/workspace/r1/.c1.patch');
  assert.match(sandbox.calls.at(-1).script, /^git apply --3way --whitespace=nowarn '\/workspace\/r1\/\.c1\.patch'$/);
  assert.deepEqual(sandbox.deletes, ['/workspace/r1/.c1.patch']);
});

test('a patch that does not apply reports ok false with the git error', async () => {
  const sandbox = createFakeSandbox({
    files: withBaseSha('r1', 'c1'),
    handlers: [{ match: /git apply/, result: { exitCode: 1, stderr: 'error: patch failed\n' } }],
  });
  const api = makeApi(sandbox);
  const response = await api.handle(request('POST', '/v1/runs/r1/worktrees/c1/patch', {
    body: { diff: 'diff --git a/x b/x\n' },
  }));
  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), { ok: false, error: 'error: patch failed' });
});

/* --------------------------------------------------------------- commit */

test('commit stages the worktree commits and points the candidate branch', async () => {
  const sha = '2222222222222222222222222222222222222222';
  const sandbox = createFakeSandbox({
    files: withBaseSha('r1', 'c1'),
    handlers: [{ match: /^git rev-parse HEAD$/, result: { stdout: `${sha}\n` } }],
  });
  const api = makeApi(sandbox);
  const response = await api.handle(request('POST', '/v1/runs/r1/worktrees/c1/commit', {
    body: { message: "candidate's work" },
  }));
  assert.deepEqual(await response.json(), { sha, branch: 'kolm/code/r1/c1' });
  const scripts = sandbox.calls.map((c) => c.script);
  assert.ok(scripts.includes('git add -A'));
  assert.ok(scripts.some((s) => s.includes("commit -q -m 'candidate'\\''s work' --allow-empty-message")));
  assert.ok(scripts.some((s) => s === `git branch -f 'kolm/code/r1/c1' '${sha}'`));
  for (const call of sandbox.calls) assert.equal(call.cwd, '/workspace/r1/c1');
});

test('a failing commit surfaces the git error as an unprocessable response', async () => {
  const sandbox = createFakeSandbox({
    files: withBaseSha('r1', 'c1'),
    handlers: [{ match: /commit -q/, result: { exitCode: 1, stderr: 'nothing to commit\n' } }],
  });
  const api = makeApi(sandbox);
  const response = await api.handle(request('POST', '/v1/runs/r1/worktrees/c1/commit', { body: { message: 'x' } }));
  assert.equal(response.status, 422);
  assert.deepEqual(await response.json(), { error: 'commit-failed', message: 'nothing to commit' });
});

/* -------------------------------------------------------------- destroy */

test('destroying a worktree removes it prunes and drops its metadata', async () => {
  const sandbox = createFakeSandbox({ files: withBaseSha('r1', 'c1') });
  const api = makeApi(sandbox);
  const response = await api.handle(request('DELETE', '/v1/runs/r1/worktrees/c1'));
  assert.deepEqual(await response.json(), { ok: true });
  const scripts = sandbox.calls.map((c) => c.script);
  assert.ok(scripts[0].includes("worktree remove --force '/workspace/r1/c1'"));
  assert.ok(scripts[0].includes("rm -rf '/workspace/r1/c1'"), 'removal falls back to rm when git refuses');
  assert.ok(scripts.some((s) => s.includes('worktree prune')));
  assert.deepEqual(sandbox.deletes, ['/workspace/r1/c1.base']);
});

test('destroying a run clears the workspace and stops the container', async () => {
  const sandbox = createFakeSandbox({ files: withBaseSha('r1', 'c1') });
  const api = makeApi(sandbox);
  const response = await api.handle(request('DELETE', '/v1/runs/r1'));
  assert.deepEqual(await response.json(), { ok: true });
  assert.equal(sandbox.calls[0].script, "rm -rf '/workspace/r1'");
  assert.equal(sandbox.destroyed, 1);
});

/* --------------------------------------------------------------- routing */

test('unknown routes and methods are reported without touching the container', async () => {
  const sandbox = createFakeSandbox({ files: withBaseSha('r1', 'c1') });
  const api = makeApi(sandbox);
  const notFound = await api.handle(request('GET', '/v1/nope'));
  assert.equal(notFound.status, 404);
  const badMethod = await api.handle(request('PATCH', '/v1/runs/r1/worktrees'));
  assert.equal(badMethod.status, 405);
  const badChild = await api.handle(request('GET', '/v1/runs/r1/worktrees/c1/unknown'));
  assert.equal(badChild.status, 404);
  assert.equal(sandbox.calls.length, 0);
});

test('a malformed json body is rejected with a client error', async () => {
  const sandbox = createFakeSandbox({ files: withBaseSha('r1', 'c1') });
  const api = makeApi(sandbox);
  const req = new Request('https://sandbox.example/v1/runs/r1/worktrees/c1/exec', {
    method: 'POST',
    headers: { authorization: `Bearer ${TOKEN}`, 'content-type': 'application/json' },
    body: '{not json',
  });
  const response = await api.handle(req);
  assert.equal(response.status, 400);
  assert.equal((await response.json()).error, 'invalid-body');
});

/* ------------------------------------------------------- egress allowlist */

test('the container allowlist covers the source and package hosts the ladder needs', () => {
  for (const host of ['github.com', 'codeload.github.com', 'objects.githubusercontent.com',
    'registry.npmjs.org', 'pypi.org', 'files.pythonhosted.org']) {
    assert.ok(DEFAULT_ALLOWED_HOSTS.includes(host), `${host} must be reachable`);
  }
  assert.deepEqual(parseAllowedHosts(' proxy.internal , , bad host, dup.example,dup.example '),
    ['proxy.internal', 'dup.example']);
  assert.deepEqual(parseAllowedHosts(undefined), []);
});
