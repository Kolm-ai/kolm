// Owner C — command detection, tool-output parsers, repo brief and baseline (spec 5.5).
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { detectCommands, detectRunner, parseTestOutput, parseTypecheckErrors, parseLintErrors, relativize } from '../src/ladder/context/detect.js';
import { repoBrief, briefToPrompt, BRIEF_BYTE_CAP } from '../src/ladder/context/repo-brief.js';
import { baseline, runCommand } from '../src/ladder/context/baseline.js';
import { openStore, sqliteAvailable } from '../src/ladder/store.js';

// The cache assertions need node:sqlite; on Node 20 they skip instead of failing.
const HAS_SQLITE = sqliteAvailable();
const needsStore = HAS_SQLITE ? {} : { skip: 'node:sqlite is unavailable on this Node' };

function tempDir(label) {
  return fs.mkdtempSync(path.join(os.tmpdir(), `kolm-ctx-${label}-`));
}

function writeFiles(root, files) {
  for (const [rel, body] of Object.entries(files)) {
    const target = path.join(root, rel);
    fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.writeFileSync(target, body);
  }
  return root;
}

function walk(root, prefix = '') {
  const out = [];
  for (const entry of fs.readdirSync(path.join(root, prefix), { withFileTypes: true })) {
    const rel = prefix ? `${prefix}/${entry.name}` : entry.name;
    if (entry.isDirectory()) out.push(...walk(root, rel));
    else out.push(rel);
  }
  return out;
}

/** Inline stand-in for owner B's worktree; only the five methods stages use. */
function fakeWorktree(root, execHandler) {
  const counters = { readFile: 0, listFiles: 0, exec: 0, diff: 0 };
  return {
    id: 'wt-fake',
    path: root,
    counters,
    async readFile(rel) {
      counters.readFile += 1;
      return fs.readFileSync(path.join(root, rel), 'utf8');
    },
    async listFiles({ limit = 1000 } = {}) {
      counters.listFiles += 1;
      return walk(root).slice(0, limit);
    },
    async exec(cmd, opts = {}) {
      counters.exec += 1;
      const res = execHandler ? execHandler(cmd, opts) : null;
      return { stdout: '', stderr: '', exitCode: 0, durationMs: 1, timedOut: false, truncated: false, ...(res || {}) };
    },
    async diff() {
      counters.diff += 1;
      return '';
    },
    async changedFiles() {
      return [];
    },
    async baseSha() {
      return 'sha-fake';
    },
  };
}

// ------------------------------------------------------------ detectCommands

test('detectCommands reads a Node repo with scripts tsconfig and eslint', async () => {
  const root = writeFiles(tempDir('node'), {
    'package.json': JSON.stringify({
      name: 'demo',
      scripts: { test: 'node --test', build: 'tsc' },
      devDependencies: { c8: '^9.0.0', eslint: '^9.0.0' },
    }),
    'tsconfig.json': '{"compilerOptions":{"strict":true}}',
    'eslint.config.js': 'export default [];',
    'package-lock.json': '{}',
    'src/index.ts': 'export const x = 1;',
  });
  const commands = await detectCommands(fakeWorktree(root));
  assert.equal(commands.language, 'typescript');
  assert.equal(commands.packageManager, 'npm');
  assert.equal(commands.test, 'npm test --silent');
  assert.equal(commands.typecheck, 'npx tsc --noEmit -p .');
  assert.equal(commands.lint, 'npx eslint . -f json');
  assert.equal(commands.coverage, 'npx c8 --reporter=json-summary --reporter=text npm test --silent');
  fs.rmSync(root, { recursive: true, force: true });
});

test('detectCommands falls back to the declared runner and lockfile', async () => {
  const root = writeFiles(tempDir('vitest'), {
    'package.json': JSON.stringify({ name: 'demo', devDependencies: { vitest: '^1.0.0' } }),
    'pnpm-lock.yaml': 'lockfileVersion: 6.0',
    'src/index.js': 'export const x = 1;',
  });
  const commands = await detectCommands(fakeWorktree(root));
  assert.equal(commands.language, 'javascript');
  assert.equal(commands.packageManager, 'pnpm');
  assert.equal(commands.test, 'npx vitest run');
  assert.equal(commands.typecheck, null);
  assert.equal(commands.lint, null);
  assert.equal(commands.coverage, null);
  fs.rmSync(root, { recursive: true, force: true });
});

test('detectCommands reads a Python project from pyproject', async () => {
  const root = writeFiles(tempDir('py'), {
    'pyproject.toml': [
      '[project]',
      'name = "demo"',
      '[tool.mypy]',
      'strict = true',
      '[tool.ruff]',
      'line-length = 100',
    ].join('\n'),
    'demo/__init__.py': '',
  });
  const commands = await detectCommands(fakeWorktree(root));
  assert.equal(commands.language, 'python');
  assert.equal(commands.test, 'pytest -q');
  assert.equal(commands.typecheck, 'mypy .');
  assert.equal(commands.lint, 'ruff check --output-format json');
  fs.rmSync(root, { recursive: true, force: true });
});

test('detectCommands reads Go and Rust manifests', async () => {
  const goRoot = writeFiles(tempDir('go'), { 'go.mod': 'module example.com/demo\n\ngo 1.22\n', 'main.go': 'package main\n' });
  const go = await detectCommands(fakeWorktree(goRoot));
  assert.equal(go.language, 'go');
  assert.equal(go.test, 'go test ./...');
  assert.equal(go.typecheck, 'go vet ./...');
  assert.equal(go.coverage, 'go test ./... -cover');
  fs.rmSync(goRoot, { recursive: true, force: true });

  const rustRoot = writeFiles(tempDir('rs'), { 'Cargo.toml': '[package]\nname = "demo"\n', 'src/main.rs': 'fn main() {}\n' });
  const rust = await detectCommands(fakeWorktree(rustRoot));
  assert.equal(rust.language, 'rust');
  assert.equal(rust.test, 'cargo test');
  assert.equal(rust.typecheck, 'cargo check');
  fs.rmSync(rustRoot, { recursive: true, force: true });
});

test('detectCommands returns nulls for a repo it cannot classify', async () => {
  const root = writeFiles(tempDir('bare'), { 'notes.txt': 'nothing to see' });
  const commands = await detectCommands(fakeWorktree(root));
  assert.deepEqual(commands, { test: null, typecheck: null, lint: null, coverage: null, language: null, packageManager: null });
  fs.rmSync(root, { recursive: true, force: true });
});

test('detectRunner maps a command string to a parser hint', () => {
  assert.equal(detectRunner('node --test tests/'), 'node');
  assert.equal(detectRunner('npm test --silent'), 'node');
  assert.equal(detectRunner('npx vitest run'), 'vitest');
  assert.equal(detectRunner('npx jest --ci'), 'jest');
  assert.equal(detectRunner('pytest -q'), 'pytest');
  assert.equal(detectRunner('go test ./...'), 'go');
  assert.equal(detectRunner('cargo test'), 'cargo');
  assert.equal(detectRunner(''), 'unknown');
  assert.equal(detectRunner('make check'), 'unknown');
});

// ----------------------------------------------------------- parseTestOutput

test('parseTestOutput reads the node spec reporter', () => {
  const out = [
    '\u2714 adds two numbers (1.23ms)',
    '\u2714 keeps the sign (0.51ms)',
    '\u2716 divides by zero (2.04ms)',
    '\u2139 tests 3',
    '\u2139 pass 2',
    '\u2139 fail 1',
  ].join('\n');
  const parsed = parseTestOutput(out, 'node');
  assert.equal(parsed.parsed, true);
  assert.equal(parsed.passed, 2);
  assert.equal(parsed.failed, 1);
  assert.deepEqual(parsed.failing, ['divides by zero']);
});

test('parseTestOutput reads node TAP output with hash counters', () => {
  const out = [
    'TAP version 13',
    'ok 1 - alpha works',
    'not ok 2 - beta fails',
    '  ---',
    '  duration_ms: 1.2',
    '  ...',
    '1..2',
    '# pass 1',
    '# fail 1',
  ].join('\n');
  const parsed = parseTestOutput(out, 'node');
  assert.equal(parsed.passed, 1);
  assert.equal(parsed.failed, 1);
  assert.deepEqual(parsed.failing, ['beta fails']);
});

test('parseTestOutput reads jest and vitest summaries', () => {
  const jest = parseTestOutput([
    '  \u2713 renders header (5 ms)',
    '  \u2715 submits the form (7 ms)',
    'Tests:       1 failed, 3 passed, 4 total',
  ].join('\n'), 'jest');
  assert.equal(jest.passed, 3);
  assert.equal(jest.failed, 1);
  assert.deepEqual(jest.failing, ['submits the form']);

  const vitest = parseTestOutput([
    ' \u2713 src/a.test.js > works',
    ' \u2717 src/b.test.js > breaks',
    ' Tests  1 failed | 3 passed (4)',
  ].join('\n'), 'vitest');
  assert.equal(vitest.passed, 3);
  assert.equal(vitest.failed, 1);
  assert.deepEqual(vitest.failing, ['src/b.test.js > breaks']);
});

test('parseTestOutput reads pytest failures and its summary line', () => {
  const out = [
    'FAILED tests/test_math.py::test_add - assert 3 == 4',
    'ERROR tests/test_io.py::test_read',
    '==================== 1 failed, 3 passed in 0.12s ====================',
  ].join('\n');
  const parsed = parseTestOutput(out, 'pytest');
  assert.equal(parsed.passed, 3);
  assert.equal(parsed.failed, 1);
  assert.deepEqual(parsed.failing, ['tests/test_math.py::test_add', 'tests/test_io.py::test_read']);
});

test('parseTestOutput reads go test and cargo test output', () => {
  const go = parseTestOutput([
    '--- FAIL: TestAdd (0.00s)',
    '--- PASS: TestSub (0.00s)',
    'FAIL\texample/pkg\t0.012s',
    'ok  \texample/other\t0.005s',
  ].join('\n'), 'go');
  assert.equal(go.failed, 1);
  assert.deepEqual(go.failing, ['TestAdd']);
  assert.equal(go.passed, 1);

  const cargo = parseTestOutput([
    'test tests::adds ... ok',
    'test tests::subs ... FAILED',
    'test result: FAILED. 1 passed; 1 failed; 0 ignored',
  ].join('\n'), 'cargo');
  assert.equal(cargo.passed, 1);
  assert.equal(cargo.failed, 1);
  assert.deepEqual(cargo.failing, ['tests::subs']);
});

test('parseTestOutput reports parsed false on unrecognisable output', () => {
  const parsed = parseTestOutput('Segmentation fault (core dumped)', 'node');
  assert.equal(parsed.parsed, false);
  assert.equal(parsed.passed, 0);
  assert.equal(parsed.failed, 0);
  assert.deepEqual(parsed.failing, []);
  assert.equal(parseTestOutput(null).parsed, false);
});

// ------------------------------------------------------- error id extraction

test('parseTypecheckErrors gives tsc errors line-independent ids', () => {
  const before = parseTypecheckErrors(
    "src/a.ts(12,5): error TS2345: Argument of type 'string' is not assignable to parameter of type 'number'.",
    'tsc',
  );
  assert.equal(before.length, 1);
  assert.match(before[0], /^src\/a\.ts:TS2345:[0-9a-f]{8}$/);
  const shifted = parseTypecheckErrors(
    "src/a.ts(48,9): error TS2345: Argument of type 'string' is not assignable to parameter of type 'number'.",
    'tsc',
  );
  assert.deepEqual(shifted, before, 'a line shift must not look like a new error');
  const different = parseTypecheckErrors('src/a.ts(12,5): error TS2345: Something else entirely.', 'tsc');
  assert.notDeepEqual(different, before);
});

test('parseTypecheckErrors strips the worktree root from paths', () => {
  const root = 'C:/tmp/wt-7';
  const ids = parseTypecheckErrors('C:/tmp/wt-7/src/deep/a.ts(3,1): error TS1005: comma expected.', 'tsc', { root });
  assert.match(ids[0], /^src\/deep\/a\.ts:TS1005:/);
  assert.equal(relativize('C:\\tmp\\wt-7\\src\\a.ts', root), 'src/a.ts');
  assert.equal(relativize('./src/a.ts'), 'src/a.ts');
});

test('parseTypecheckErrors reads pyright json and mypy text', () => {
  const pyright = parseTypecheckErrors(JSON.stringify({
    generalDiagnostics: [
      { file: '/repo/app/main.py', severity: 'error', rule: 'reportGeneralTypeIssues', message: 'x is not int' },
      { file: '/repo/app/main.py', severity: 'warning', rule: 'reportUnusedImport', message: 'unused' },
    ],
  }), 'pyright', { root: '/repo' });
  assert.equal(pyright.length, 1, 'warnings do not gate a patch');
  assert.match(pyright[0], /^app\/main\.py:reportGeneralTypeIssues:[0-9a-f]{8}$/);

  const mypy = parseTypecheckErrors('app/main.py:14: error: Incompatible return value type  [return-value]', 'mypy');
  assert.equal(mypy.length, 1);
  assert.match(mypy[0], /^app\/main\.py:return-value:[0-9a-f]{8}$/);
});

test('parseLintErrors reads eslint json and skips warnings', () => {
  const payload = JSON.stringify([
    {
      filePath: '/repo/src/a.js',
      messages: [
        { ruleId: 'no-undef', severity: 2, line: 10, column: 3, message: 'x is not defined' },
        { ruleId: 'no-console', severity: 1, line: 12, column: 1, message: 'unexpected console' },
      ],
    },
    { filePath: '/repo/src/b.js', messages: [] },
  ]);
  const ids = parseLintErrors(payload, 'eslint', { root: '/repo' });
  assert.deepEqual(ids, ['src/a.js:no-undef:10']);
  assert.deepEqual(parseLintErrors('[]', 'eslint'), []);
});

test('parseLintErrors reads ruff json output', () => {
  const payload = JSON.stringify([
    { filename: '/repo/app/main.py', code: 'F401', location: { row: 3, column: 1 }, message: 'unused import' },
    { filename: '/repo/app/main.py', code: 'E501', location: { row: 40, column: 1 }, message: 'line too long' },
  ]);
  assert.deepEqual(parseLintErrors(payload, 'ruff', { root: '/repo' }), ['app/main.py:F401:3', 'app/main.py:E501:40']);
});

// ------------------------------------------------------------------- brief

test('repoBrief stays inside the byte cap and briefToPrompt does too', async () => {
  const root = tempDir('brief');
  const files = { 'README.md': `# Demo\n${'lorem ipsum dolor sit amet '.repeat(200)}` };
  files['package.json'] = JSON.stringify({ name: 'big', version: '1.0.0', scripts: { test: 'node --test' } });
  for (let i = 0; i < 600; i += 1) {
    files[`src/module-with-a-fairly-long-name-${i}/index.js`] = `export function handler${i}() { return ${i}; }\n`;
  }
  writeFiles(root, files);
  const wt = fakeWorktree(root);
  const brief = await repoBrief(null, wt, 'sha-brief');
  assert.equal(brief.sha, 'sha-brief');
  assert.ok(brief.tree.length > 0);
  assert.ok(brief.tree.length <= 400, 'the tree is capped at 400 entries');
  assert.ok(brief.bytes <= BRIEF_BYTE_CAP, `brief is ${brief.bytes} bytes`);
  assert.equal(brief.languages['.js'] > 0, true);
  assert.equal(brief.manifests['package.json'].name, 'big');
  assert.equal(brief.commands.test, 'npm test --silent');
  const prompt = briefToPrompt(brief);
  assert.ok(Buffer.byteLength(prompt, 'utf8') <= BRIEF_BYTE_CAP);
  assert.ok(prompt.includes('REPO BRIEF'));
  assert.equal(briefToPrompt(null), '');
  fs.rmSync(root, { recursive: true, force: true });
});

test('repoBrief serves the second call from the context cache', needsStore, async () => {
  const root = writeFiles(tempDir('briefcache'), {
    'package.json': JSON.stringify({ name: 'cached', scripts: { test: 'node --test' } }),
    'src/a.js': 'export function alpha() {}\nexport const beta = 2;\n',
    'README.md': '# Cached\n',
  });
  const dbDir = tempDir('briefdb');
  const store = openStore(path.join(dbDir, 'ladder.db'));
  try {
    const wt = fakeWorktree(root);
    const first = await repoBrief(store, wt, 'sha-1');
    const after = { ...wt.counters };
    assert.ok(after.listFiles >= 1);
    assert.deepEqual(first.exports['src/a.js'], ['alpha', 'beta']);

    const second = await repoBrief(store, wt, 'sha-1');
    assert.deepEqual(second, first);
    assert.deepEqual(wt.counters, after, 'a cache hit does no worktree work');

    const other = await repoBrief(store, wt, 'sha-2');
    assert.equal(other.sha, 'sha-2');
    assert.ok(wt.counters.listFiles > after.listFiles, 'a new sha rebuilds the brief');
  } finally {
    store.close();
    fs.rmSync(root, { recursive: true, force: true });
    fs.rmSync(dbDir, { recursive: true, force: true });
  }
});

// ---------------------------------------------------------------- baseline

const BASELINE_COMMANDS = {
  test: 'npm test --silent',
  typecheck: 'npx tsc --noEmit -p .',
  lint: 'npx eslint . -f json',
  coverage: null,
};

function baselineExec(cmd) {
  if (cmd.includes('tsc')) {
    return { stdout: 'src/old.ts(3,1): error TS2322: Type mismatch that predates the task.\n', exitCode: 2, durationMs: 120 };
  }
  if (cmd.includes('eslint')) {
    return {
      stdout: JSON.stringify([{ filePath: 'src/old.js', messages: [{ ruleId: 'no-unused-vars', severity: 2, line: 7, message: 'unused' }] }]),
      exitCode: 1,
      durationMs: 90,
    };
  }
  return {
    stdout: ['\u2714 already green (1ms)', '\u2716 already red (1ms)', '\u2139 pass 1', '\u2139 fail 1'].join('\n'),
    exitCode: 1,
    durationMs: 240,
  };
}

test('baseline records pre-existing failures errors and durations', async () => {
  const root = writeFiles(tempDir('base'), { 'package.json': '{}' });
  const wt = fakeWorktree(root, baselineExec);
  const result = await baseline(null, wt, 'sha-base', BASELINE_COMMANDS, { config: { stageTimeoutMs: 5000 } });
  assert.equal(result.sha, 'sha-base');
  assert.deepEqual(result.tests.failing, ['already red']);
  assert.equal(result.tests.passed, 1);
  assert.equal(result.tests.exitCode, 1);
  assert.equal(result.tests.runner, 'node');
  assert.equal(result.typecheck.errors.length, 1);
  assert.match(result.typecheck.errors[0], /^src\/old\.ts:TS2322:/);
  assert.deepEqual(result.lint.errors, ['src/old.js:no-unused-vars:7']);
  assert.equal(result.durations.typecheck, 120);
  assert.equal(result.durations.lint, 90);
  assert.equal(result.durations.test, 240);
  assert.equal(wt.counters.exec, 3);
  fs.rmSync(root, { recursive: true, force: true });
});

test('baseline skips commands the repo does not have', async () => {
  const root = writeFiles(tempDir('basenone'), { 'package.json': '{}' });
  const wt = fakeWorktree(root, baselineExec);
  const result = await baseline(null, wt, 'sha-none', { test: null, typecheck: null, lint: null });
  assert.equal(wt.counters.exec, 0);
  assert.equal(result.tests.ran, false);
  assert.deepEqual(result.tests.failing, []);
  assert.deepEqual(result.typecheck.errors, []);
  fs.rmSync(root, { recursive: true, force: true });
});

test('baseline caches per sha and reuses the stored measurement', needsStore, async () => {
  const root = writeFiles(tempDir('basecache'), { 'package.json': '{}' });
  const dbDir = tempDir('basedb');
  const store = openStore(path.join(dbDir, 'ladder.db'));
  try {
    const wt = fakeWorktree(root, baselineExec);
    const first = await baseline(store, wt, 'sha-1', BASELINE_COMMANDS);
    assert.equal(wt.counters.exec, 3);
    const second = await baseline(store, wt, 'sha-1', BASELINE_COMMANDS);
    assert.deepEqual(second, first);
    assert.equal(wt.counters.exec, 3, 'a cache hit runs nothing');
    await baseline(store, wt, 'sha-1', BASELINE_COMMANDS, { force: true });
    assert.equal(wt.counters.exec, 6, 'force re-measures');
  } finally {
    store.close();
    fs.rmSync(root, { recursive: true, force: true });
    fs.rmSync(dbDir, { recursive: true, force: true });
  }
});

test('runCommand turns an exec throw into a recorded failure', async () => {
  const root = writeFiles(tempDir('throw'), { 'package.json': '{}' });
  const wt = fakeWorktree(root, () => {
    throw new Error('spawn ENOENT');
  });
  const res = await runCommand(wt, 'npm test --silent', { timeoutMs: 1000 });
  assert.equal(res.ran, true);
  assert.equal(res.exitCode, null);
  assert.match(res.output, /exec failed: spawn ENOENT/);
  const none = await runCommand(wt, null, {});
  assert.equal(none.ran, false);
  assert.equal(none.output, '');
  fs.rmSync(root, { recursive: true, force: true });
});
