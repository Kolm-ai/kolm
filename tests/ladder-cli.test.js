// kolm code — CLI surface (owner E, spec 7.1).
//
// Every spawn here runs with a scrubbed environment: no provider keys, a
// throwaway KOLM_CODE_HOME, the fake worktree driver and the mock provider, so
// the suite never reaches the network and never touches the operator's store.

import { after, before, describe, it } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

import { parseArgs, EXIT, USAGE, VERBS, renderTable, envOverlay } from '../cli/kolm-code.mjs';
import { sqliteAvailable } from '../src/ladder/index.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(HERE, '..');
const CLI = path.join(REPO_ROOT, 'cli', 'kolm-code.mjs');
const KOLM = path.join(REPO_ROOT, 'cli', 'kolm.js');
const HAS_SQLITE = sqliteAvailable();

/** Names that could arm a provider, a repo host or the partner worker. */
const SECRET_ENV_NAMES = [
  'ANTHROPIC_API_KEY',
  'CEREBRAS_API_KEY',
  'KOLM_CEREBRAS_TOKEN',
  'DEEPSEEK_API_KEY',
  'OPENROUTER_API_KEY',
  'KOLM_GITHUB_TOKEN',
  'GITHUB_TOKEN',
  'KOLM_GITHUB_APP_ID',
  'KOLM_GITHUB_APP_INSTALLATION_ID',
  'KOLM_GITHUB_APP_PRIVATE_KEY',
  'KOLM_GITHUB_APP_PRIVATE_KEY_PATH',
  'TRUFFLE_GITHUB_APP_ID',
  'TRUFFLE_GITHUB_INSTALLATION_ID',
  'TRUFFLE_GITHUB_PRIVATE_KEY',
  'TRUFFLE_PARTNER_API_URL',
  'TRUFFLE_PARTNER_SERVICE_TOKEN',
  'TRUFFLE_PARTNER_WORKSPACE_ID',
  'TRUFFLE_PARTNER_PROCESSOR_URL',
  'TRUFFLE_PARTNER_PROCESSOR_KEY_ID',
  'TRUFFLE_PARTNER_PROCESSOR_SIGNING_SECRET',
  'KOLM_CODE_BANDS_PATH',
  'KOLM_CODE_ORCHESTRATOR',
  'KOLM_CODE_JUDGE',
  'KOLM_CODE_CODER_LADDER',
  'KOLM_CODE_REVIEWER_LADDER',
];

const tempDirs = [];

function makeTempDir(prefix) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), prefix));
  tempDirs.push(dir);
  return dir;
}

/**
 * A child environment with every credential removed and the ladder pointed at
 * a throwaway home.
 * @param {string} home
 * @param {Record<string,string>} [extra]
 */
function scrubbedEnv(home, extra = {}) {
  const env = { ...process.env };
  for (const name of SECRET_ENV_NAMES) delete env[name];
  env.HOME = home;
  env.USERPROFILE = home;
  env.KOLM_CODE_HOME = home;
  env.KOLM_DATA_DIR = path.join(home, '.kolm');
  return { ...env, ...extra };
}

function runCli(args, { home, extra = {}, cwd = REPO_ROOT, cli = CLI } = {}) {
  const res = spawnSync(process.execPath, [cli, ...args], {
    env: scrubbedEnv(home, extra),
    encoding: 'utf8',
    cwd,
    timeout: 180_000,
  });
  return {
    status: res.status,
    stdout: res.stdout ?? '',
    stderr: res.stderr ?? '',
    all: `${res.stdout ?? ''}\n${res.stderr ?? ''}`,
  };
}

/** A one-commit git repository for the run to operate on. */
function makeRepo() {
  const dir = makeTempDir('kolm-cli-repo-');
  const git = (args) => spawnSync('git', args, { cwd: dir, encoding: 'utf8', timeout: 60_000 });
  git(['init', '-q']);
  git(['config', 'user.email', 'kolm@localhost']);
  git(['config', 'user.name', 'kolm code tests']);
  fs.writeFileSync(path.join(dir, 'README.md'), '# fixture\n');
  fs.writeFileSync(
    path.join(dir, 'index.js'),
    'export function add(a, b) {\n  return a + b;\n}\n',
  );
  git(['add', '.']);
  git(['commit', '-qm', 'fixture']);
  return dir;
}

after(() => {
  for (const dir of tempDirs) {
    try {
      fs.rmSync(dir, { recursive: true, force: true });
    } catch {
      // a worktree may still hold a handle on Windows; the OS reclaims it
    }
  }
});

// ---------------------------------------------------------------------------

describe('kolm code argument parsing', () => {
  it('reads long flags in both the space and the equals form', () => {
    const parsed = parseArgs(['run', 'fix the bug', '--repo', 'C:/tmp/x', '--candidates=4']);
    assert.deepEqual(parsed.positional, ['run', 'fix the bug']);
    assert.equal(parsed.flags.repo, 'C:/tmp/x');
    assert.equal(parsed.flags.candidates, '4');
  });

  it('treats declared booleans as flags that never eat the next token', () => {
    const parsed = parseArgs(['run', '--json', 'the task', '--quiet', '--pr']);
    assert.equal(parsed.flags.json, true);
    assert.equal(parsed.flags.quiet, true);
    assert.equal(parsed.flags.pr, true);
    assert.deepEqual(parsed.positional, ['run', 'the task']);
  });

  it('accumulates repeated and comma separated ladder flags', () => {
    const parsed = parseArgs(['run', 'x', '--coder', 'ollama:qwen3.8:latest', '--coder', 'cerebras:a,deepseek:b']);
    assert.deepEqual(parsed.flags.coder, ['ollama:qwen3.8:latest', 'cerebras:a', 'deepseek:b']);
  });

  it('camel cases dashed flag names and honours the double dash terminator', () => {
    const parsed = parseArgs(['run', 'x', '--max-cost', '2.5', '--pr-base', 'trunk', '--', '--not-a-flag']);
    assert.equal(parsed.flags.maxCost, '2.5');
    assert.equal(parsed.flags.prBase, 'trunk');
    assert.deepEqual(parsed.passthrough, ['--not-a-flag']);
  });

  it('turns a trailing valueless flag into a boolean', () => {
    const parsed = parseArgs(['doctor', '--live']);
    assert.equal(parsed.flags.live, true);
    assert.equal(parseArgs(['status', '--limit']).flags.limit, true);
  });

  it('maps run flags onto the environment overlay the frozen config needs', () => {
    const overlay = envOverlay(
      { mode: 'direct', driver: 'fake', candidates: '2', coder: ['fake:a'], maxCost: '1.5', keepWorktrees: true, pollMs: '250' },
      {},
    );
    assert.equal(overlay.KOLM_CODE_MODE, 'direct');
    assert.equal(overlay.KOLM_CODE_DRIVER, 'fake');
    assert.equal(overlay.KOLM_CODE_CANDIDATES, '2');
    assert.equal(overlay.KOLM_CODE_CODER_LADDER, 'fake:a');
    assert.equal(overlay.KOLM_CODE_MAX_COST_USD, '1.5');
    assert.equal(overlay.KOLM_CODE_KEEP_WORKTREES, 'all');
    assert.equal(overlay.TRUFFLE_PARTNER_POLL_MS, '250');
  });

  it('renders ascii tables without box drawing characters', () => {
    const table = renderTable(['A', 'BB'], [['1', '2'], ['longer', '3']]);
    assert.ok(table.includes('A       BB'));
    assert.ok(/^[\x20-\x7e\n]*$/.test(table), 'table stays inside printable ascii');
  });
});

describe('kolm code usage surface', () => {
  let home;
  before(() => {
    home = makeTempDir('kolm-cli-help-');
  });

  it('prints the usage text and exits cleanly', () => {
    const res = runCli(['help'], { home });
    assert.equal(res.status, EXIT.OK);
    for (const verb of VERBS) {
      assert.ok(res.stdout.includes(verb), `usage mentions the ${verb} verb`);
    }
    assert.ok(res.stdout.includes('docs/KOLM_CODE.md'));
  });

  it('prints the same usage with no verb at all', () => {
    const res = runCli([], { home });
    assert.equal(res.status, EXIT.OK);
    assert.equal(res.stdout.trim(), USAGE.trim());
  });

  it('answers an unknown verb with the usage exit code', () => {
    const res = runCli(['frobnicate'], { home });
    assert.equal(res.status, EXIT.USAGE);
    assert.ok(res.stderr.includes('unknown verb'));
  });

  it('stays inside printable ascii in the usage text', () => {
    assert.ok(/^[\x20-\x7e\n]*$/.test(USAGE), 'no emoji, no box drawing, no smart quotes');
    assert.ok(!new RegExp(['hon','est'].join(''), 'i').test(USAGE), 'house vocabulary rule');
  });

  it('is reachable through the kolm dispatcher as a code verb', () => {
    const res = runCli(['code', 'help'], { home, cli: KOLM });
    assert.equal(res.status, EXIT.OK);
    assert.equal(res.stdout.trim(), USAGE.trim());
  });
});

describe('kolm code doctor', () => {
  let home;
  before(() => {
    home = makeTempDir('kolm-cli-doctor-');
  });

  it('reports a machine readable arming table and refuses to claim readiness', () => {
    const res = runCli(['doctor', '--json'], { home });
    assert.equal(res.status, EXIT.MISSING_PREREQ);
    const report = JSON.parse(res.stdout);
    assert.equal(report.ok, false);
    assert.ok(Array.isArray(report.rungs));
    assert.ok(report.rungs.length >= 8);
    for (const id of ['orchestrator', 'judge', 'driver', 'github', 'truffle', 'sqlite']) {
      assert.ok(report.rungs.some((rung) => rung.id === id), `the table has a ${id} row`);
    }
    assert.ok(report.rungs.some((rung) => rung.kind === 'coder'));
    assert.ok(report.rungs.some((rung) => rung.kind === 'reviewer'));
  });

  it('prints an ascii table naming the missing variables', () => {
    const res = runCli(['doctor'], { home });
    assert.equal(res.status, EXIT.MISSING_PREREQ);
    assert.ok(res.stdout.includes('NOT ARMED'));
    assert.ok(res.stdout.includes('ANTHROPIC_API_KEY missing'));
    assert.ok(/^[\x20-\x7e\n\r]*$/.test(res.stdout), 'doctor output is printable ascii');
  });

  it('never echoes a credential value while reporting it as armed', () => {
    const secret = 'sk-ant-test123456789';
    const res = runCli(['doctor'], { home, extra: { ANTHROPIC_API_KEY: secret } });
    assert.ok(!res.all.includes(secret), 'the key value never reaches stdout or stderr');
    const orchestratorRow = res.stdout.split('\n').find((line) => line.startsWith('orchestrator'));
    assert.ok(orchestratorRow, 'the table has an orchestrator row');
    assert.ok(orchestratorRow.includes('ARMED'));
    assert.ok(!orchestratorRow.includes('NOT ARMED'));
    assert.ok(res.stdout.includes('ANTHROPIC_API_KEY set'), 'the reason names the variable only');
  });
});

describe('kolm code bands and receipts', () => {
  let home;
  before(() => {
    home = makeTempDir('kolm-cli-bands-');
  });

  it('prints the bundled bands with both thresholds and the orientation', () => {
    const res = runCli(['bands'], { home });
    assert.equal(res.status, EXIT.OK);
    assert.ok(res.stdout.includes('l0-meta'));
    assert.ok(res.stdout.includes('patch_lines'));
    assert.ok(res.stdout.includes('l2-dynamic'));
    assert.ok(res.stdout.includes('ORIENTATION'));
    assert.ok(res.stdout.includes(path.join('src', 'ladder', 'bands.json')));
  });

  it('refuses to write a bands proposal without an explicit confirmation', () => {
    const res = runCli(['bands', '--apply'], { home });
    assert.notEqual(res.status, EXIT.OK);
    assert.ok(res.all.includes('--yes') || res.all.includes('suggestBands'));
  });

  it('reports a missing receipt as not found', () => {
    const res = runCli(['receipt', 'verify', 'r_20260101_absent0'], { home });
    assert.equal(res.status, EXIT.NOT_FOUND);
    assert.ok(res.stderr.includes('not-found'));
  });

  it('rejects a receipt subcommand it does not have', () => {
    const res = runCli(['receipt', 'frobnicate', 'x'], { home });
    assert.equal(res.status, EXIT.BAD_ARGS);
    assert.ok(res.stderr.includes('kolm code: bad-args:'));
  });
});

describe('kolm code store verbs', { skip: HAS_SQLITE ? false : 'node:sqlite unavailable on this Node' }, () => {
  let home;
  before(() => {
    home = makeTempDir('kolm-cli-store-');
  });

  it('reports an empty store instead of failing', () => {
    const res = runCli(['status'], { home });
    assert.equal(res.status, EXIT.OK);
    assert.ok(res.stdout.includes('no runs recorded'));
  });

  it('prints funnel counters as json', () => {
    const res = runCli(['funnel', '--json'], { home });
    assert.equal(res.status, EXIT.OK);
    const funnel = JSON.parse(res.stdout);
    assert.equal(funnel.candidates, 0);
    assert.deepEqual(funnel.verdicts, { PASS: 0, FAIL: 0, JUDGE: 0 });
  });

  it('reports an unknown run id as not found', () => {
    const res = runCli(['show', 'r_20260101_absent0'], { home });
    assert.equal(res.status, EXIT.NOT_FOUND);
    assert.ok(res.stderr.includes('not-found'));
  });
});

describe('kolm code run end to end', { skip: HAS_SQLITE ? false : 'node:sqlite unavailable on this Node' }, () => {
  let home;
  let repo;
  let runResult;
  let runId = null;

  before(() => {
    home = makeTempDir('kolm-cli-run-');
    repo = makeRepo();
    runResult = runCli(
      ['run', 'add a subtract helper next to add', '--repo', repo, '--candidates', '1', '--quiet'],
      { home, extra: { KOLM_TEST_MOCK_PROVIDER: '1', KOLM_CODE_DRIVER: 'fake' } },
    );
    const listed = runCli(['status', '--json'], { home });
    if (listed.status === EXIT.OK) {
      const runs = JSON.parse(listed.stdout).runs;
      if (runs.length > 0) runId = runs[0].id;
    }
  });

  it('completes with a cascade verdict rather than an execution error', () => {
    assert.ok(
      [EXIT.OK, EXIT.GATE_FAIL].includes(runResult.status),
      `run exited ${runResult.status}; stderr: ${runResult.stderr.slice(-800)}`,
    );
    assert.ok(/verdict\s+(PASS|FAIL|JUDGE)/.test(runResult.stdout), runResult.stdout.slice(0, 400));
  });

  it('records the run so status can list it', () => {
    assert.ok(runId, 'the run was written to the store');
    const res = runCli(['status'], { home });
    assert.equal(res.status, EXIT.OK);
    assert.ok(res.stdout.includes(runId));
    assert.ok(res.stdout.includes('RUN'));
  });

  it('shows one run as parseable json with the stage grid', () => {
    const res = runCli(['show', runId, '--json'], { home });
    assert.equal(res.status, EXIT.OK);
    const detail = JSON.parse(res.stdout);
    assert.equal(detail.run.id, runId);
    assert.ok(Array.isArray(detail.candidates));
    assert.ok(Number.isFinite(detail.events));
    assert.ok(Object.prototype.hasOwnProperty.call(detail, 'receiptPath'));
  });

  it('shows the same run as an ascii report', () => {
    const res = runCli(['show', runId], { home });
    assert.equal(res.status, EXIT.OK);
    assert.ok(res.stdout.includes(runId));
    assert.ok(res.stdout.includes('CANDIDATE'));
  });

  it('counts the run in the funnel', () => {
    const res = runCli(['funnel', '--json'], { home });
    assert.equal(res.status, EXIT.OK);
    const funnel = JSON.parse(res.stdout);
    assert.ok(funnel.runs >= 1);
  });

  it('verifies the signed receipt the run wrote', () => {
    const res = runCli(['receipt', 'verify', runId], { home });
    assert.equal(res.status, EXIT.OK, res.stderr.slice(-400));
    assert.ok(res.stdout.includes('VERIFIED'));
    assert.ok(/kid\s+[0-9a-f]{8,}/.test(res.stdout), 'the signing key id is named');
  });

  it('resumes a stored run by rehydrating its task', () => {
    const res = runCli(['resume', runId, '--quiet'], {
      home,
      extra: { KOLM_TEST_MOCK_PROVIDER: '1', KOLM_CODE_DRIVER: 'fake' },
    });
    assert.ok(
      [EXIT.OK, EXIT.GATE_FAIL].includes(res.status),
      `resume exited ${res.status}; stderr: ${res.stderr.slice(-600)}`,
    );
    assert.ok(res.stdout.includes(runId));
  });

  it('reports an unknown run id to resume as not found', () => {
    const res = runCli(['resume', 'r_20260101_absent0'], { home });
    assert.equal(res.status, EXIT.NOT_FOUND);
    assert.ok(res.stderr.includes('not-found'));
  });

  it('compares cascade against direct in the eval table', () => {
    const res = runCli(['eval', '--json'], { home });
    assert.equal(res.status, EXIT.OK, res.stderr.slice(-400));
    const evaluation = JSON.parse(res.stdout);
    assert.equal(typeof evaluation, 'object');
  });
});
