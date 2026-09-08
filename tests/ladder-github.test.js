// kolm code — owner F tests: GitHub App auth + pull requests (spec section 9).
//
// No network and no child processes: every boundary is injected
// (`fetchImpl`, `execImpl`, `readFile`, `exists`, `home`, `now`).

import test from 'node:test';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import os from 'node:os';
import path from 'node:path';

import {
  DEFAULT_APP_KEY_RELATIVE_PATH,
  _clearTokenCache,
  buildAppJwt,
  describeGitHubArming,
  getInstallationToken,
  githubEnvFromConfig,
  loadPrivateKey,
  redactSecrets,
  resolveGitHubToken,
  resolvePrivateKeySource,
} from '../src/ladder/github/app-auth.js';
import {
  buildPrBody,
  detectRemote,
  openPrForRun,
  openPullRequest,
  pushBranch,
  remoteToOwnerRepo,
} from '../src/ladder/github/pr.js';

const keyPair = crypto.generateKeyPairSync('rsa', {
  modulusLength: 2048,
  publicKeyEncoding: { type: 'spki', format: 'pem' },
  privateKeyEncoding: { type: 'pkcs8', format: 'pem' },
});
const PRIVATE_PEM = keyPair.privateKey;
const PUBLIC_PEM = keyPair.publicKey;
const FAKE_HOME = path.join(os.tmpdir(), 'kolm-code-owner-f-home');
const NEVER_EXISTS = () => false;
const NEVER_READ = () => {
  throw new Error('readFile must not be called in this case');
};

function jsonResponse(status, payload, { ok } = {}) {
  const text = typeof payload === 'string' ? payload : JSON.stringify(payload);
  return {
    ok: ok ?? (status >= 200 && status < 300),
    status,
    text: async () => text,
  };
}

function caught(fn) {
  try {
    fn();
  } catch (err) {
    return err;
  }
  return assert.fail('expected the call to throw');
}

function recordingFetch(handler) {
  const calls = [];
  const impl = async (url, options) => {
    calls.push({ url, options });
    return handler(url, options, calls.length);
  };
  impl.calls = calls;
  return impl;
}

/* ------------------------------------------------------------------ *
 * private key resolution
 * ------------------------------------------------------------------ */

test('private key prefers the inline Truffle env var and un-escapes newlines', () => {
  const escaped = PRIVATE_PEM.replace(/\n/g, '\\n');
  const resolved = resolvePrivateKeySource({
    env: { TRUFFLE_GITHUB_PRIVATE_KEY: escaped },
    readFile: NEVER_READ,
    exists: NEVER_EXISTS,
    home: FAKE_HOME,
  });
  assert.equal(resolved.source, 'TRUFFLE_GITHUB_PRIVATE_KEY');
  assert.equal(resolved.path, null);
  assert.equal(resolved.pem, PRIVATE_PEM);
  assert.ok(resolved.pem.includes('\n'));
  assert.ok(!resolved.pem.includes('\\n'));
});

test('private key falls through path env vars in the documented order', () => {
  const seen = [];
  const readFile = (file) => {
    seen.push(file);
    return PRIVATE_PEM;
  };
  const kolmFirst = resolvePrivateKeySource({
    env: {
      KOLM_GITHUB_APP_PRIVATE_KEY_PATH: '/keys/kolm.pem',
      TRUFFLE_GITHUB_PRIVATE_KEY_PATH: '/keys/truffle.pem',
    },
    readFile,
    exists: NEVER_EXISTS,
    home: FAKE_HOME,
  });
  assert.equal(kolmFirst.source, 'KOLM_GITHUB_APP_PRIVATE_KEY_PATH');
  assert.equal(seen.length, 1);

  const truffleNext = resolvePrivateKeySource({
    env: { TRUFFLE_GITHUB_PRIVATE_KEY_PATH: '/keys/truffle.pem' },
    readFile,
    exists: NEVER_EXISTS,
    home: FAKE_HOME,
  });
  assert.equal(truffleNext.source, 'TRUFFLE_GITHUB_PRIVATE_KEY_PATH');
});

test('private key uses the default Truffle path only when that file exists', () => {
  const expected = path.join(FAKE_HOME, ...DEFAULT_APP_KEY_RELATIVE_PATH.split('/'));
  const checked = [];
  const missing = loadPrivateKey({
    env: {},
    readFile: NEVER_READ,
    exists: (file) => {
      checked.push(file);
      return false;
    },
    home: FAKE_HOME,
  });
  assert.equal(missing, null);
  assert.deepEqual(checked, [expected]);

  const present = loadPrivateKey({
    env: {},
    readFile: (file) => {
      assert.equal(file, expected);
      return PRIVATE_PEM;
    },
    exists: (file) => file === expected,
    home: FAKE_HOME,
  });
  assert.equal(present, PRIVATE_PEM);
});

test('private key read failures never quote key material', () => {
  const err = caught(() => loadPrivateKey({
    env: { KOLM_GITHUB_APP_PRIVATE_KEY_PATH: '/keys/missing.pem' },
    readFile: () => {
      const failure = new Error(`ENOENT reading ${PRIVATE_PEM}`);
      failure.code = 'ENOENT';
      throw failure;
    },
    exists: NEVER_EXISTS,
    home: FAKE_HOME,
  }));
  assert.equal(err.code, 'github-app-key');
  assert.ok(!err.message.includes('PRIVATE KEY'));
  assert.ok(!JSON.stringify(err.detail).includes('PRIVATE KEY'));
});

test('redaction removes pem blocks, tokens and jwts from any text', () => {
  const text = `${PRIVATE_PEM} ghp_${'a'.repeat(36)} secret-value`;
  const clean = redactSecrets(text, ['secret-value']);
  assert.ok(!clean.includes('PRIVATE KEY'));
  assert.ok(!clean.includes('ghp_'));
  assert.ok(!clean.includes('secret-value'));
});

/* ------------------------------------------------------------------ *
 * App JWT
 * ------------------------------------------------------------------ */

test('app jwt verifies under the matching public key with the documented claims', () => {
  const now = 1_700_000_000_000;
  const jwt = buildAppJwt({ appId: '123456', privateKey: PRIVATE_PEM, now: () => now });
  const [header, claims, signature] = jwt.split('.');
  assert.equal(signature.length > 0, true);

  const verified = crypto
    .createVerify('RSA-SHA256')
    .update(`${header}.${claims}`)
    .verify(PUBLIC_PEM, Buffer.from(signature, 'base64url'));
  assert.equal(verified, true);

  assert.deepEqual(JSON.parse(Buffer.from(header, 'base64url').toString('utf8')), {
    alg: 'RS256',
    typ: 'JWT',
  });
  const payload = JSON.parse(Buffer.from(claims, 'base64url').toString('utf8'));
  assert.equal(payload.iss, '123456');
  assert.equal(payload.iat, Math.floor(now / 1000) - 60);
  assert.equal(payload.exp, Math.floor(now / 1000) + 540);
});

test('app jwt signing failures report a reason and never the key', () => {
  // A real key of the wrong kind: RS256 cannot sign with Ed25519, so signing
  // fails with key material in hand, which is exactly what must not leak.
  const wrongKind = crypto.generateKeyPairSync('ed25519', {
    privateKeyEncoding: { type: 'pkcs8', format: 'pem' },
    publicKeyEncoding: { type: 'spki', format: 'pem' },
  });
  const err = caught(() => buildAppJwt({
    appId: '1',
    privateKey: wrongKind.privateKey,
  }));
  assert.equal(err.code, 'github-app-auth');
  assert.ok(!err.message.includes('PRIVATE KEY'));
  assert.ok(!JSON.stringify(err.detail ?? {}).includes('PRIVATE KEY'));
});

/* ------------------------------------------------------------------ *
 * installation tokens
 * ------------------------------------------------------------------ */

test('installation token exchange sends the documented request', async () => {
  _clearTokenCache();
  const fetchImpl = recordingFetch(() => jsonResponse(201, {
    token: 'ghs_installation_token',
    expires_at: new Date(1_700_000_000_000 + 3_600_000).toISOString(),
    permissions: { contents: 'write', pull_requests: 'write' },
  }));

  const issued = await getInstallationToken({
    appId: '123456',
    installationId: '42',
    privateKey: PRIVATE_PEM,
    fetchImpl,
    now: () => 1_700_000_000_000,
    apiBase: 'https://api.github.test',
  });

  assert.equal(issued.token, 'ghs_installation_token');
  assert.deepEqual(issued.permissions, { contents: 'write', pull_requests: 'write' });

  const [call] = fetchImpl.calls;
  assert.equal(call.url, 'https://api.github.test/app/installations/42/access_tokens');
  assert.equal(call.options.method, 'POST');
  assert.match(call.options.headers.authorization, /^Bearer eyJ/);
  assert.equal(call.options.headers.accept, 'application/vnd.github+json');
  assert.equal(call.options.headers['x-github-api-version'], '2022-11-28');
  assert.equal(call.options.headers['user-agent'], 'kolm-code');
});

test('installation token is cached until the refresh margin, then re-fetched', async () => {
  _clearTokenCache();
  const start = 1_700_000_000_000;
  const fetchImpl = recordingFetch((_url, _options, n) => jsonResponse(201, {
    token: `ghs_token_${n}`,
    expires_at: new Date(start + 120_000).toISOString(),
    permissions: {},
  }));

  const first = await getInstallationToken({
    appId: 'app', installationId: 'inst', privateKey: PRIVATE_PEM, fetchImpl, now: () => start,
  });
  const cached = await getInstallationToken({
    appId: 'app', installationId: 'inst', privateKey: PRIVATE_PEM, fetchImpl, now: () => start + 30_000,
  });
  assert.equal(first.token, 'ghs_token_1');
  assert.equal(cached.token, 'ghs_token_1');
  assert.equal(fetchImpl.calls.length, 1);

  // 30 s before expiry is inside the 60 s refresh margin, so a new token issues.
  const refreshed = await getInstallationToken({
    appId: 'app', installationId: 'inst', privateKey: PRIVATE_PEM, fetchImpl, now: () => start + 90_000,
  });
  assert.equal(refreshed.token, 'ghs_token_2');
  assert.equal(fetchImpl.calls.length, 2);

  const other = await getInstallationToken({
    appId: 'app', installationId: 'other', privateKey: PRIVATE_PEM, fetchImpl, now: () => start,
  });
  assert.equal(other.token, 'ghs_token_3');
});

test('installation token failures carry status and a redacted excerpt', async () => {
  _clearTokenCache();
  let sentJwt = null;
  const fetchImpl = async (_url, options) => {
    sentJwt = String(options.headers.authorization).replace('Bearer ', '');
    return jsonResponse(
      401,
      `{"message":"Bad credentials for ${sentJwt}","documentation_url":"https://docs.github.com"}`,
    );
  };

  const thrown = await getInstallationToken({
    appId: 'app', installationId: 'inst', privateKey: PRIVATE_PEM, fetchImpl, now: () => 1,
  }).then(() => null, (e) => e);
  assert.equal(thrown.code, 'github-app-auth');
  assert.match(thrown.message, /401/);
  assert.equal(thrown.detail.status, 401);
  assert.ok(thrown.detail.body.length <= 300);
  assert.ok(!thrown.detail.body.includes(sentJwt));
  assert.ok(!thrown.message.includes('PRIVATE KEY'));
});

/* ------------------------------------------------------------------ *
 * token resolution ladder
 * ------------------------------------------------------------------ */

test('token resolution follows KOLM_GITHUB_TOKEN then GITHUB_TOKEN', async () => {
  const both = await resolveGitHubToken({
    env: { KOLM_GITHUB_TOKEN: 'kolm-token', GITHUB_TOKEN: 'generic-token' },
    fetchImpl: () => assert.fail('no network expected'),
    execImpl: () => assert.fail('no exec expected'),
    readFile: NEVER_READ,
    exists: NEVER_EXISTS,
    home: FAKE_HOME,
  });
  assert.deepEqual({ token: both.token, source: both.source }, {
    token: 'kolm-token', source: 'KOLM_GITHUB_TOKEN',
  });

  const generic = await resolveGitHubToken({
    env: { GITHUB_TOKEN: 'generic-token' },
    fetchImpl: () => assert.fail('no network expected'),
    execImpl: () => assert.fail('no exec expected'),
    readFile: NEVER_READ,
    exists: NEVER_EXISTS,
    home: FAKE_HOME,
  });
  assert.equal(generic.source, 'GITHUB_TOKEN');
});

test('token resolution uses the app when it is fully configured', async () => {
  _clearTokenCache();
  const fetchImpl = recordingFetch(() => jsonResponse(201, {
    token: 'ghs_app_token',
    expires_at: new Date(2_000_000_000_000).toISOString(),
    permissions: { contents: 'write' },
  }));
  const resolved = await resolveGitHubToken({
    env: {
      KOLM_GITHUB_APP_ID: '99',
      KOLM_GITHUB_APP_INSTALLATION_ID: '7',
      TRUFFLE_GITHUB_PRIVATE_KEY: PRIVATE_PEM,
    },
    fetchImpl,
    execImpl: () => assert.fail('gh must not run when the app succeeds'),
    readFile: NEVER_READ,
    exists: NEVER_EXISTS,
    home: FAKE_HOME,
    now: () => 1_700_000_000_000,
  });
  assert.equal(resolved.source, 'github-app');
  assert.equal(resolved.token, 'ghs_app_token');
  assert.equal(fetchImpl.calls.length, 1);
});

test('token resolution falls through a failing app to the gh cli', async () => {
  _clearTokenCache();
  const tried = [];
  const resolved = await resolveGitHubToken({
    env: {
      KOLM_GITHUB_APP_ID: '99',
      KOLM_GITHUB_APP_INSTALLATION_ID: '7',
      TRUFFLE_GITHUB_PRIVATE_KEY: PRIVATE_PEM,
    },
    fetchImpl: async () => jsonResponse(500, '{"message":"server error"}'),
    execImpl: (file, args) => {
      tried.push([file, args.join(' ')]);
      if (file === 'gh') throw new Error('ENOENT');
      return 'gho_cli_token\n';
    },
    readFile: NEVER_READ,
    exists: NEVER_EXISTS,
    home: FAKE_HOME,
    now: () => 1,
  });
  assert.equal(resolved.source, 'gh-cli');
  assert.equal(resolved.token, 'gho_cli_token');
  assert.deepEqual(tried, [['gh', 'auth token'], ['gh.cmd', 'auth token']]);
  assert.ok(resolved.reasons.some((reason) => reason.startsWith('github-app:')));
  assert.ok(!JSON.stringify(resolved.reasons).includes('PRIVATE KEY'));
});

test('token resolution returns null when nothing is armed', async () => {
  const resolved = await resolveGitHubToken({
    env: {},
    fetchImpl: () => assert.fail('no network expected'),
    execImpl: () => {
      throw new Error('gh not installed');
    },
    readFile: NEVER_READ,
    exists: NEVER_EXISTS,
    home: FAKE_HOME,
  });
  assert.equal(resolved, null);
});

test('arming report is pure and reports the cli as unknown', () => {
  const empty = describeGitHubArming({
    env: {}, readFile: NEVER_READ, exists: NEVER_EXISTS, home: FAKE_HOME,
  });
  assert.equal(empty.armed, false);
  assert.equal(empty.ghCli, 'unknown');
  assert.ok(empty.missing.includes('KOLM_GITHUB_TOKEN'));
  assert.deepEqual(empty.sources, []);

  const armed = describeGitHubArming({
    env: {
      KOLM_GITHUB_APP_ID: '99',
      KOLM_GITHUB_APP_INSTALLATION_ID: '7',
      TRUFFLE_GITHUB_PRIVATE_KEY: PRIVATE_PEM,
    },
    readFile: NEVER_READ,
    exists: NEVER_EXISTS,
    home: FAKE_HOME,
  });
  assert.equal(armed.armed, true);
  assert.equal(armed.privateKeySource, 'TRUFFLE_GITHUB_PRIVATE_KEY');
  assert.ok(armed.sources.includes('github-app'));
});

test('the config adapter projects owner A config onto the github env names', async () => {
  _clearTokenCache();
  const config = {
    github: {
      appId: '99',
      installationId: '7',
      tokenSource: 'GITHUB_TOKEN',
      privateKeySource: 'TRUFFLE_GITHUB_PRIVATE_KEY',
    },
    getSecret: (name) => {
      assert.equal(name, 'githubToken');
      return 'config-token';
    },
    githubPrivateKeyPem: () => PRIVATE_PEM,
  };

  const env = githubEnvFromConfig(config);
  assert.equal(env.GITHUB_TOKEN, 'config-token');
  assert.equal(env.KOLM_GITHUB_APP_ID, '99');
  assert.equal(env.KOLM_GITHUB_APP_PRIVATE_KEY, PRIVATE_PEM);

  const resolved = await resolveGitHubToken({
    config,
    fetchImpl: () => assert.fail('no network expected'),
    execImpl: () => assert.fail('no exec expected'),
    readFile: NEVER_READ,
    exists: NEVER_EXISTS,
    home: FAKE_HOME,
  });
  assert.deepEqual({ token: resolved.token, source: resolved.source }, {
    token: 'config-token', source: 'GITHUB_TOKEN',
  });

  const arming = describeGitHubArming({
    config, readFile: NEVER_READ, exists: NEVER_EXISTS, home: FAKE_HOME,
  });
  assert.equal(arming.armed, true);
  assert.equal(arming.privateKeySource, 'KOLM_GITHUB_APP_PRIVATE_KEY');
  assert.deepEqual(githubEnvFromConfig(null), {});
});

/* ------------------------------------------------------------------ *
 * pull requests
 * ------------------------------------------------------------------ */

test('pull request creation posts the documented payload', async () => {
  const fetchImpl = recordingFetch(() => jsonResponse(201, {
    number: 12,
    url: 'https://api.github.test/repos/o/r/pulls/12',
    html_url: 'https://github.com/o/r/pull/12',
  }));
  const pr = await openPullRequest({
    owner: 'o',
    repo: 'r',
    head: 'kolm/run-1',
    base: 'main',
    title: 'kolm code: fix the parser',
    body: 'body text',
    token: 'ghs_token',
    fetchImpl,
    apiBase: 'https://api.github.test',
  });
  assert.deepEqual(pr, {
    number: 12,
    url: 'https://api.github.test/repos/o/r/pulls/12',
    htmlUrl: 'https://github.com/o/r/pull/12',
    created: true,
  });
  const [call] = fetchImpl.calls;
  assert.equal(call.url, 'https://api.github.test/repos/o/r/pulls');
  assert.deepEqual(JSON.parse(call.options.body), {
    title: 'kolm code: fix the parser',
    head: 'kolm/run-1',
    base: 'main',
    body: 'body text',
    draft: true,
  });
  assert.equal(call.options.headers['user-agent'], 'kolm-code');
});

test('an existing pull request is reused instead of failing', async () => {
  const fetchImpl = recordingFetch((url) => {
    if (url.endsWith('/pulls')) {
      return jsonResponse(422, {
        message: 'Validation Failed',
        errors: [{ message: 'A pull request already exists for o:kolm/run-1.' }],
      });
    }
    return jsonResponse(200, [{
      number: 5,
      url: 'https://api.github.test/repos/o/r/pulls/5',
      html_url: 'https://github.com/o/r/pull/5',
    }]);
  });
  const pr = await openPullRequest({
    owner: 'o', repo: 'r', head: 'kolm/run-1', title: 't', token: 'ghs_token',
    fetchImpl, apiBase: 'https://api.github.test',
  });
  assert.deepEqual(pr, {
    number: 5,
    url: 'https://api.github.test/repos/o/r/pulls/5',
    htmlUrl: 'https://github.com/o/r/pull/5',
    created: false,
  });
  assert.equal(fetchImpl.calls.length, 2);
  assert.ok(fetchImpl.calls[1].url.includes('head=o%3Akolm%2Frun-1'));
  assert.ok(fetchImpl.calls[1].url.includes('state=open'));
});

test('other pull request failures raise a ladder error with an excerpt', async () => {
  const long = 'y'.repeat(600);
  const fetchImpl = async () => jsonResponse(
    403,
    `{"message":"Resource not accessible by integration ghs_secret_token","x":"${long}"}`,
  );
  const thrown = await openPullRequest({
    owner: 'o', repo: 'r', head: 'b', title: 't', token: 'ghs_secret_token', fetchImpl,
  }).then(() => null, (e) => e);
  assert.equal(thrown.code, 'github-pr');
  assert.match(thrown.message, /403/);
  assert.equal(thrown.detail.status, 403);
  assert.ok(thrown.detail.body.length <= 300);
  assert.ok(!thrown.detail.body.includes('ghs_secret_token'));
});

test('missing token is reported as a not-armed failure', async () => {
  const thrown = await openPullRequest({
    owner: 'o', repo: 'r', head: 'b', title: 't', token: null,
    fetchImpl: () => assert.fail('no network expected'),
  }).then(() => null, (e) => e);
  assert.equal(thrown.code, 'not-armed');
});

/* ------------------------------------------------------------------ *
 * push
 * ------------------------------------------------------------------ */

test('branch push injects the token as a one-shot http extraheader', () => {
  const calls = [];
  const result = pushBranch({
    repoPath: '/work/tree',
    branch: 'kolm/run-1',
    token: 'ghs_push_token',
    execImpl: (file, args, options) => {
      calls.push({ file, args, options });
      return '';
    },
  });
  assert.deepEqual(result, { pushed: true, remote: 'origin', branch: 'kolm/run-1' });

  const [call] = calls;
  assert.equal(call.file, 'git');
  const expected = Buffer.from('x-access-token:ghs_push_token', 'utf8').toString('base64');
  assert.deepEqual(call.args, [
    '-c',
    `http.extraheader=AUTHORIZATION: basic ${expected}`,
    'push',
    'origin',
    'kolm/run-1:refs/heads/kolm/run-1',
  ]);
  assert.equal(call.options.cwd, '/work/tree');
  assert.equal(call.options.timeout, 120_000);
});

test('branch push redacts the token and its base64 blob from stderr', () => {
  const token = 'ghs_push_token';
  const basic = Buffer.from(`x-access-token:${token}`, 'utf8').toString('base64');
  const thrown = caught(() => pushBranch({
    repoPath: '/work/tree',
    branch: 'kolm/run-1',
    token,
    execImpl: () => {
      const err = new Error(`Command failed: git -c http.extraheader=AUTHORIZATION: basic ${basic} push`);
      err.status = 128;
      err.stderr = `remote: Invalid credentials ${token}\nfatal: Authentication failed`;
      throw err;
    },
  }));
  assert.equal(thrown.code, 'git-push');
  assert.equal(thrown.detail.status, 128);
  assert.ok(thrown.detail.stderr.includes('[redacted]'));
  assert.ok(!thrown.detail.stderr.includes(token));
  assert.ok(!thrown.detail.stderr.includes(basic));
  assert.ok(!thrown.message.includes(token));
});

/* ------------------------------------------------------------------ *
 * remotes, body, orchestration
 * ------------------------------------------------------------------ */

test('remote urls parse in all three github spellings', () => {
  assert.deepEqual(remoteToOwnerRepo('https://github.com/kolm-ai/kolm.git'), {
    owner: 'kolm-ai', repo: 'kolm',
  });
  assert.deepEqual(remoteToOwnerRepo('git@github.com:kolm-ai/kolm.git'), {
    owner: 'kolm-ai', repo: 'kolm',
  });
  assert.deepEqual(remoteToOwnerRepo('ssh://git@github.com/kolm-ai/kolm'), {
    owner: 'kolm-ai', repo: 'kolm',
  });
  assert.equal(remoteToOwnerRepo('https://gitlab.com/kolm-ai/kolm.git'), null);
  assert.equal(remoteToOwnerRepo(''), null);
  assert.equal(remoteToOwnerRepo(null), null);
});

test('remote detection shells out to git and tolerates failure', () => {
  const url = detectRemote({
    repoPath: '/work/tree',
    execImpl: (file, args) => {
      assert.equal(file, 'git');
      assert.deepEqual(args, ['remote', 'get-url', 'origin']);
      return 'https://github.com/kolm-ai/kolm.git\n';
    },
  });
  assert.equal(url, 'https://github.com/kolm-ai/kolm.git');

  const missing = detectRemote({
    repoPath: '/work/tree',
    execImpl: () => {
      throw new Error('no such remote');
    },
  });
  assert.equal(missing, null);
});

test('pull request body carries the funnel, the receipt and the verify line', () => {
  const receipt = {
    run_id: 'run-abc',
    task: { title: 'fix the parser' },
    winner: { candidate: 'cand-2', branch: 'kolm/run-abc' },
    verdict: 'pass',
    cost: { total_usd: 0.1234 },
    stages: [
      { stage: 'L0', decision: 'pass', reasons: [] },
      { stage: 'L1', decision: 'fail', reasons: ['lint: 2 errors'] },
    ],
  };
  const body = buildPrBody({ receipt, receiptPath: '~/.kolm/code/runs/run-abc/receipt.json' });
  assert.ok(body.includes('## kolm code: fix the parser'));
  assert.ok(body.includes('- winner candidate: cand-2'));
  assert.ok(body.includes('- cost: $0.1234'));
  assert.ok(body.includes('| stage | decision | reasons |'));
  assert.ok(body.includes('| L1 | fail | lint: 2 errors |'));
  assert.ok(body.includes('```json'));
  assert.ok(body.includes('"run_id": "run-abc"'));
  assert.ok(body.includes('Verify: kolm code receipt verify ~/.kolm/code/runs/run-abc/receipt.json'));
  assert.ok(body.trimEnd().endsWith('Generated by kolm code'));

  const objectStages = buildPrBody({ receipt: { stages: { L0: { decision: 'pass' }, L2: 'skip' } } });
  assert.ok(objectStages.includes('| L0 | pass |'));
  assert.ok(objectStages.includes('| L2 | skip |'));

  const bare = buildPrBody({});
  assert.ok(bare.includes('(no stage results recorded)'));
  assert.ok(bare.includes('Verify: kolm code receipt verify receipt.json'));
});

test('pull request body never carries a credential', () => {
  const body = buildPrBody({
    receipt: {
      task: { title: 'leaky' },
      env: { token: `ghp_${'b'.repeat(36)}`, pem: PRIVATE_PEM },
    },
  });
  assert.ok(!body.includes('PRIVATE KEY'));
  assert.ok(!body.includes('ghp_'));
});

test('run orchestration detects the remote, pushes, then opens the request', async () => {
  const exec = [];
  const fetchImpl = recordingFetch(() => jsonResponse(201, {
    number: 3,
    url: 'https://api.github.test/repos/kolm-ai/kolm/pulls/3',
    html_url: 'https://github.com/kolm-ai/kolm/pull/3',
  }));
  const result = await openPrForRun({
    repoPath: '/work/tree',
    branch: 'kolm/run-abc',
    base: 'main',
    title: 'kolm code: fix the parser',
    body: 'body',
    token: 'ghs_token',
    fetchImpl,
    apiBase: 'https://api.github.test',
    execImpl: (file, args) => {
      exec.push(args);
      if (args[0] === 'remote') return 'git@github.com:kolm-ai/kolm.git\n';
      return '';
    },
  });
  assert.deepEqual(result, {
    number: 3,
    url: 'https://api.github.test/repos/kolm-ai/kolm/pulls/3',
    htmlUrl: 'https://github.com/kolm-ai/kolm/pull/3',
    pushed: true,
    created: true,
    owner: 'kolm-ai',
    repo: 'kolm',
    branch: 'kolm/run-abc',
  });
  assert.equal(exec.length, 2);
  assert.deepEqual(exec[0], ['remote', 'get-url', 'origin']);
  assert.equal(exec[1][exec[1].length - 1], 'kolm/run-abc:refs/heads/kolm/run-abc');
});

test('run orchestration refuses to guess a non-github remote', async () => {
  const thrown = await openPrForRun({
    repoPath: '/work/tree',
    branch: 'b',
    token: 'ghs_token',
    fetchImpl: () => assert.fail('no network expected'),
    execImpl: () => 'https://example.invalid/x/y.git\n',
  }).then(() => null, (e) => e);
  assert.equal(thrown.code, 'github-pr');
  assert.match(thrown.message, /owner\/repo/);
});
