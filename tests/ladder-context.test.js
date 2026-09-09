// Owner C — command detection, tool-output parsers, repo brief and baseline (spec 5.5).
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { detectCommands, detectRunner, parseTestOutput, parseTypecheckErrors, parseLintErrors, relativize, makefileTargets, parsePnpmWorkspaces } from '../src/ladder/context/detect.js';
import { repoBrief, briefToPrompt, repoKeyFor, parseExports, testDirsFrom, BRIEF_BYTE_CAP } from '../src/ladder/context/repo-brief.js';
import { retrieveContext, retrievalToPrompt, extractTerms, searchPatternFor, toAscii, isVendored } from '../src/ladder/context/retrieve.js';
import { loadInstructions, instructionsToPrompt, INSTRUCTIONS_BYTE_CAP } from '../src/ladder/context/instructions.js';
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
  const counters = { readFile: 0, listFiles: 0, search: 0, exec: 0, diff: 0 };
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
    // Same contract as the fake driver's grep: JS RegExp, {path, line, text}.
    async search(pattern, { limit = 200 } = {}) {
      counters.search += 1;
      const re = new RegExp(pattern);
      const out = [];
      for (const rel of walk(root)) {
        if (out.length >= limit) break;
        let text;
        try {
          text = fs.readFileSync(path.join(root, rel), 'utf8');
        } catch {
          continue;
        }
        const lines = text.split(/\r?\n/);
        for (let i = 0; i < lines.length && out.length < limit; i += 1) {
          if (re.test(lines[i])) out.push({ path: rel, line: i + 1, text: lines[i] });
        }
      }
      return out;
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
  assert.deepEqual(commands, {
    test: null, typecheck: null, lint: null, coverage: null,
    language: null, packageManager: null, monorepo: false, workspaces: [],
  });
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
  assert.equal(brief.languages.js > 0, true, 'languages are keyed by language name, not extension');
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

// ------------------------------------------------- detect upgrades (spec 1.4)

test('detectCommands uses Makefile targets only when no manifest answered', async () => {
  const root = writeFiles(tempDir('make'), {
    'Makefile': '.PHONY: test lint\n\ntest:\n\tgcc -o t t.c && ./t\n\ncheck:\n\tcppcheck .\n\nlint:\n\tclang-tidy *.c\n',
    'main.c': 'int main(void) { return 0; }\n',
  });
  const commands = await detectCommands(fakeWorktree(root));
  assert.equal(commands.test, 'make test');
  assert.equal(commands.typecheck, 'make check');
  assert.equal(commands.lint, 'make lint');
  assert.equal(commands.packageManager, 'make');
  fs.rmSync(root, { recursive: true, force: true });
});

test('detectCommands keeps the manifest answer over a Makefile target', async () => {
  const root = writeFiles(tempDir('makenode'), {
    'package.json': JSON.stringify({ name: 'both', scripts: { test: 'node --test' } }),
    'Makefile': 'test:\n\techo wrong\n',
  });
  const commands = await detectCommands(fakeWorktree(root));
  assert.equal(commands.test, 'npm test --silent', 'a package.json script wins over make test');
  fs.rmSync(root, { recursive: true, force: true });
});

test('detectCommands reads a Deno project', async () => {
  const bare = writeFiles(tempDir('deno'), {
    'deno.json': JSON.stringify({ imports: {} }),
    'mod.ts': 'export const one = 1;\n',
  });
  const plain = await detectCommands(fakeWorktree(bare));
  assert.equal(plain.test, 'deno test');
  assert.equal(plain.lint, 'deno lint');
  assert.equal(plain.packageManager, 'deno');
  assert.equal(plain.language, 'typescript');
  fs.rmSync(bare, { recursive: true, force: true });

  const tasked = writeFiles(tempDir('denotask'), {
    'deno.json': JSON.stringify({ tasks: { test: 'deno test -A', check: 'deno check mod.ts' } }),
  });
  const commands = await detectCommands(fakeWorktree(tasked));
  assert.equal(commands.test, 'deno task test');
  assert.equal(commands.typecheck, 'deno task check');
  fs.rmSync(tasked, { recursive: true, force: true });
});

test('detectCommands reports workspaces as a monorepo', async () => {
  const npmRoot = writeFiles(tempDir('wsnpm'), {
    'package.json': JSON.stringify({ name: 'root', private: true, workspaces: ['packages/*', 'apps/web'], scripts: { test: 'node --test' } }),
  });
  const npmCommands = await detectCommands(fakeWorktree(npmRoot));
  assert.equal(npmCommands.monorepo, true);
  assert.deepEqual(npmCommands.workspaces, ['packages/*', 'apps/web']);
  fs.rmSync(npmRoot, { recursive: true, force: true });

  const pnpmRoot = writeFiles(tempDir('wspnpm'), {
    'package.json': JSON.stringify({ name: 'root' }),
    'pnpm-lock.yaml': 'lockfileVersion: 9\n',
    'pnpm-workspace.yaml': "packages:\n  - 'packages/*'\n  - 'tools/cli'\n",
  });
  const pnpmCommands = await detectCommands(fakeWorktree(pnpmRoot));
  assert.equal(pnpmCommands.monorepo, true);
  assert.deepEqual(pnpmCommands.workspaces, ['packages/*', 'tools/cli']);
  assert.equal(pnpmCommands.packageManager, 'pnpm');
  fs.rmSync(pnpmRoot, { recursive: true, force: true });
});

test('makefileTargets and parsePnpmWorkspaces read their own formats', () => {
  const targets = makefileTargets('.PHONY: test lint\n\nCC := gcc\n\ntest: build\n\t./run\n\nbuild:\n\t$(CC) x.c\n');
  assert.equal(targets.has('test'), true);
  assert.equal(targets.has('build'), true);
  assert.equal(targets.has('lint'), true);
  assert.equal(targets.has('CC'), false, 'an assignment is not a target');
  assert.deepEqual(parsePnpmWorkspaces("packages:\n  - 'a/*'\n  - \"b\"\n  - c   # trailing\nother: 1\n"), ['a/*', 'b', 'c']);
  assert.deepEqual(parsePnpmWorkspaces(''), []);
});

test('parseTestOutput reads python unittest output', () => {
  const out = [
    'test_add (tests.test_math.MathTest) ... ok',
    'test_div (tests.test_math.MathTest) ... ERROR',
    'test_skip (tests.test_math.MathTest) ... skipped \'not on windows\'',
    'test_sub (tests.test_math.MathTest) ... FAIL',
    '======================================================================',
    'FAIL: test_sub (tests.test_math.MathTest)',
    '----------------------------------------------------------------------',
    'AssertionError: 1 != 2',
    '',
    'Ran 4 tests in 0.003s',
    '',
    'FAILED (failures=1, errors=1, skipped=1)',
  ].join('\n');
  const parsed = parseTestOutput(out, 'unittest');
  assert.equal(parsed.parsed, true);
  assert.equal(parsed.passed, 1);
  assert.equal(parsed.failed, 2);
  assert.deepEqual(parsed.failing, ['tests.test_math.MathTest.test_div', 'tests.test_math.MathTest.test_sub']);

  const green = parseTestOutput('test_a (m.C) ... ok\ntest_b (m.C) ... ok\n\nRan 2 tests in 0.001s\n\nOK', 'unittest');
  assert.equal(green.passed, 2);
  assert.equal(green.failed, 0);
  assert.deepEqual(green.failing, []);
});

test('parseTestOutput reads deno test output', () => {
  const out = [
    'running 3 tests from ./mod_test.ts',
    'add ... ok (2ms)',
    'mul ... ok (1ms)',
    'sub ... FAILED (1ms)',
    ' ERRORS ',
    'sub => ./mod_test.ts:5:6',
    'error: AssertionError: Values are not equal.',
    '',
    'FAILED | 2 passed | 1 failed (5ms)',
  ].join('\n');
  const parsed = parseTestOutput(out, 'deno');
  assert.equal(parsed.parsed, true);
  assert.equal(parsed.passed, 2);
  assert.equal(parsed.failed, 1);
  assert.deepEqual(parsed.failing, ['sub']);
  assert.equal(detectRunner('deno test'), 'deno');
  assert.equal(detectRunner('python -m unittest discover -v'), 'unittest');
});

// ------------------------------------------------------- retrieval (spec 1.1)

/** A repo big enough that ranking has to do real work: 34 files. */
function retrievalRepo(label) {
  const files = {
    'package.json': JSON.stringify({ name: 'shop', scripts: { test: 'node --test' } }),
    'README.md': '# Shop\nOrder handling.\n',
    'src/pricing/discount-engine.js': [
      'const TIERS = [10, 50, 100];',
      '',
      '// Percentage taken off once an order clears a tier.',
      'export function applyBulkDiscount(order, tier) {',
      '  if (order.quantity > tier.minimum) return order.total * (1 - tier.rate);',
      '  return order.total;',
      '}',
      '',
      'export function tierFor(quantity) {',
      '  return TIERS.find((t) => quantity > t) ?? null;',
      '}',
    ].join('\n'),
    'src/ui/banner.js': [
      '// Shows the discount ribbon in the header.',
      'export function renderBanner(state) {',
      '  const label = state.discount ? "discount active" : "";',
      '  return label;',
      '}',
    ].join('\n'),
    'tests/pricing.test.js': 'import test from "node:test";\ntest("pricing", () => {});\n',
  };
  for (let i = 0; i < 30; i += 1) {
    const n = String(i).padStart(2, '0');
    files[`src/mod/widget-${n}.js`] = [
      `export function renderWidget${n}(props) {`,
      `  return { id: ${i}, props };`,
      '}',
    ].join('\n');
  }
  return writeFiles(tempDir(label), files);
}

test('retrieveContext ranks the file defining the named identifier first', async () => {
  const root = retrievalRepo('retr');
  const wt = fakeWorktree(root);
  assert.ok((await wt.listFiles({ limit: 5000 })).length >= 30, 'the fixture has 30+ files');
  const result = await retrieveContext(wt, {
    title: 'Bulk discount boundary is off by one',
    text: 'The `applyBulkDiscount` helper drops the discount when the quantity equals the tier minimum. Make the comparison inclusive.',
  });
  assert.equal(result.files[0].path, 'src/pricing/discount-engine.js');
  assert.ok(result.files[0].score > (result.files[1]?.score ?? 0), 'the defining file wins outright');
  assert.ok(result.terms.includes('applyBulkDiscount'));
  assert.ok(result.snippets.some((s) => s.path === 'src/pricing/discount-engine.js' && s.text.includes('applyBulkDiscount')));
  assert.ok(result.files.every((f) => !f.path.startsWith('src/mod/')), 'unrelated widgets score nothing');
  fs.rmSync(root, { recursive: true, force: true });
});

test('retrieveContext puts a path the task names outright at the top', async () => {
  const root = retrievalRepo('retrpath');
  const result = await retrieveContext(fakeWorktree(root), 'Rename the helper in src/mod/widget-07.js and keep its default export.');
  assert.equal(result.files[0].path, 'src/mod/widget-07.js');
  assert.match(result.files[0].why, /named in the task/);
  fs.rmSync(root, { recursive: true, force: true });
});

test('retrieveContext and retrievalToPrompt respect the byte budget', async () => {
  const root = retrievalRepo('retrcap');
  const result = await retrieveContext(fakeWorktree(root), {
    title: 'discount',
    text: 'The `applyBulkDiscount` helper and `renderBanner` both need the discount rate.',
  }, { maxBytes: 900 });
  assert.equal(result.maxBytes, 900);
  const prompt = retrievalToPrompt(result);
  assert.ok(Buffer.byteLength(prompt, 'utf8') <= 900, `prompt is ${Buffer.byteLength(prompt, 'utf8')} bytes`);
  assert.ok(prompt.startsWith('RELEVANT CODE'));
  assert.equal(/[^\x09\x0A\x20-\x7E]/.test(prompt), false, 'the retrieval block is ASCII');
  assert.equal(retrievalToPrompt(null), '');
  assert.equal(retrievalToPrompt({ files: [], snippets: [] }), '');
  fs.rmSync(root, { recursive: true, force: true });
});

test('retrieveContext returns an empty pack when there is nothing to go on', async () => {
  const root = writeFiles(tempDir('retrempty'), { 'a.js': 'export const a = 1;\n' });
  const empty = await retrieveContext(fakeWorktree(root), '   ');
  assert.deepEqual(empty.files, []);
  assert.deepEqual(empty.snippets, []);
  assert.deepEqual((await retrieveContext(null, 'anything')).files, []);
  fs.rmSync(root, { recursive: true, force: true });
});

test('extractTerms weighs structured mentions over prose and drops stop words', () => {
  const terms = extractTerms({ title: 'Fix the parser', text: 'The `parseDuration` helper in src/util/time.js returns the wrong value for "10m".' });
  const byTerm = new Map(terms.map((t) => [t.term, t.weight]));
  assert.equal(byTerm.get('parseDuration'), 1);
  assert.equal(byTerm.get('src/util/time.js'), 1);
  assert.equal(byTerm.has('the'), false, 'stop words are dropped');
  assert.equal(byTerm.has('fix'), false);
  assert.equal(byTerm.get('parser'), 0.6, 'prose is weaker than an identifier');
  assert.ok(terms.length <= 14);
});

test('searchPatternFor emits a pattern both regex dialects accept', () => {
  assert.equal(searchPatternFor('foo.js'), '[Ff][Oo][Oo]\\.[Jj][Ss]');
  assert.equal(searchPatternFor('ab'), null, 'two characters is too short to search');
  assert.equal(searchPatternFor('a b'), null, 'a space cannot be shared by both dialects');
  assert.equal(new RegExp(searchPatternFor('Widget')).test('renderWIDGET()'), true);
});

test('toAscii and isVendored keep the pack terminal-safe', () => {
  assert.equal(toAscii('a’b — c…'), "a'b - c...");
  assert.equal(toAscii(null), '');
  assert.equal(isVendored('node_modules/left-pad/index.js'), true);
  assert.equal(isVendored('src/node_modules_helper.js'), false);
});

// ---------------------------------------------------- instructions (spec 1.2)

test('loadInstructions reads the repo rule files in priority order', async () => {
  const root = writeFiles(tempDir('instr'), {
    'AGENTS.md': '# Agents\nUse pnpm, never npm.\n',
    'CONTRIBUTING.md': '# Contributing\nOne commit per change.\n',
    'README.md': '# ignored\n',
  });
  const loaded = await loadInstructions(fakeWorktree(root));
  assert.deepEqual(loaded.files.map((f) => f.path), ['AGENTS.md', 'CONTRIBUTING.md']);
  assert.ok(loaded.text.startsWith('PROJECT INSTRUCTIONS (from AGENTS.md, CONTRIBUTING.md; follow them)'));
  assert.ok(loaded.text.includes('--- AGENTS.md ---'));
  assert.ok(loaded.text.indexOf('--- AGENTS.md ---') < loaded.text.indexOf('--- CONTRIBUTING.md ---'));
  assert.ok(loaded.text.includes('Use pnpm, never npm.'));
  assert.equal(instructionsToPrompt(loaded), loaded.text);
  fs.rmSync(root, { recursive: true, force: true });
});

test('loadInstructions returns nothing when the repo ships no rules', async () => {
  const root = writeFiles(tempDir('instrnone'), { 'README.md': '# only a readme\n', 'AGENTS.md': '   \n' });
  const loaded = await loadInstructions(fakeWorktree(root));
  assert.deepEqual(loaded.files, []);
  assert.equal(loaded.text, '');
  assert.equal(instructionsToPrompt(loaded), '');
  assert.equal(instructionsToPrompt(null), '');
  assert.deepEqual((await loadInstructions(null)).files, []);
  fs.rmSync(root, { recursive: true, force: true });
});

test('loadInstructions stays inside the 6 KB budget', async () => {
  const root = writeFiles(tempDir('instrbig'), {
    'AGENTS.md': `# Agents\n${'rule about the build. '.repeat(1200)}`,
    'CLAUDE.md': `# Claude\n${'rule about the tests. '.repeat(1200)}`,
    '.cursorrules': `${'rule about the lint. '.repeat(1200)}`,
  });
  const loaded = await loadInstructions(fakeWorktree(root));
  assert.ok(Buffer.byteLength(loaded.text, 'utf8') <= INSTRUCTIONS_BYTE_CAP, `${Buffer.byteLength(loaded.text, 'utf8')} bytes`);
  assert.equal(loaded.files[0].path, 'AGENTS.md');
  assert.ok(loaded.files[0].bytes <= 3 * 1024);
  assert.ok(loaded.text.includes('[truncated]'));
  fs.rmSync(root, { recursive: true, force: true });
});

// ------------------------------------------------ repo brief upgrades (1.3)

// Pinned from the pre-1.3 implementation: a brief with none of the new fields
// must render byte for byte as it always did, or every stored prompt shifts.
const LEGACY_BRIEF = {
  sha: 'abc1234',
  generatedAt: '2026-09-09T00:00:00.000Z',
  tree: ['README.md', 'package.json', 'src/a.js', 'src/b.js', 'tests/a.test.js'],
  languages: { '.js': 3, '.md': 1, '.json': 1 },
  manifests: {
    'package.json': {
      name: 'demo',
      version: '1.0.0',
      type: 'module',
      scripts: { test: 'node --test', lint: 'eslint .' },
      dependencies: [],
      devDependencies: ['eslint'],
    },
    'go.mod': { module: 'example.com/demo', go: '1.22' },
  },
  readmeHead: '# Demo\nA small repo used to pin the brief rendering.\n',
  commands: { test: 'npm test --silent', typecheck: null, lint: 'npx eslint . -f json', coverage: null },
  exports: { 'src/a.js': ['alpha', 'beta'], 'src/b.js': ['gamma'] },
  bytes: 512,
};

const LEGACY_BRIEF_PROMPT = [
  'REPO BRIEF (base abc1234)',
  'languages: .js 3, .md 1, .json 1',
  'commands: test=npm test --silent typecheck=- lint=npx eslint . -f json coverage=-',
  'package.json: demo@1.0.0 scripts[test, lint] deps 0+1',
  'go.mod: {"module":"example.com/demo","go":"1.22"}',
  'README (head):',
  '# Demo',
  'A small repo used to pin the brief rendering.',
  '',
  'files (5 shown):',
  'README.md',
  'package.json',
  'src/a.js',
  'src/b.js',
  'tests/a.test.js',
  'exports:',
  '  src/a.js: alpha, beta',
  '  src/b.js: gamma',
].join('\n');

test('briefToPrompt renders a brief without the new fields byte for byte', () => {
  const rendered = briefToPrompt(LEGACY_BRIEF);
  assert.equal(rendered, LEGACY_BRIEF_PROMPT);
  assert.equal(Buffer.byteLength(rendered, 'utf8'), 438);
});

test('briefToPrompt renders instructions lessons and retrieval in order', () => {
  const brief = {
    ...LEGACY_BRIEF,
    instructions: { files: [{ path: 'AGENTS.md', bytes: 20 }], text: 'PROJECT INSTRUCTIONS (from AGENTS.md; follow them)\n--- AGENTS.md ---\nUse pnpm.' },
    lessons: [{ id: 'l1', text: 'Stage l1 skips in this repo: no lint command detected.', runId: null, ts: '2026-09-08T00:00:00.000Z', hits: 2 }],
    retrieval: {
      terms: ['applyBulkDiscount'],
      files: [{ path: 'src/a.js', score: 9, why: 'named in the task' }],
      snippets: [{ path: 'src/a.js', startLine: 1, endLine: 2, text: 'export function alpha() {}', score: 9 }],
      maxBytes: 4096,
    },
  };
  const rendered = briefToPrompt(brief);
  const iInstructions = rendered.indexOf('PROJECT INSTRUCTIONS');
  const iFiles = rendered.indexOf('files (5 shown):');
  const iLessons = rendered.indexOf('LESSONS FROM EARLIER RUNS IN THIS REPO');
  const iRetrieval = rendered.indexOf('RELEVANT CODE');
  assert.ok(iInstructions > 0, 'instructions are rendered');
  assert.ok(iInstructions < iFiles, 'instructions sit under the README head, above the tree');
  assert.ok(iFiles < iLessons, 'lessons follow the tree');
  assert.ok(iLessons < iRetrieval, 'retrieval is last so a byte cap cuts snippets first');
  assert.ok(rendered.startsWith('REPO BRIEF (base abc1234)'));
  assert.ok(rendered.includes('- Stage l1 skips in this repo'));
  // The core is untouched: every legacy line still appears, in order.
  let cursor = 0;
  for (const line of LEGACY_BRIEF_PROMPT.split('\n')) {
    if (!line) continue;
    const at = rendered.indexOf(line, cursor);
    assert.ok(at >= cursor, `legacy line missing or reordered: ${line}`);
    cursor = at;
  }
});

test('parseExports understands the languages the brief covers', () => {
  assert.deepEqual(parseExports('export function alpha() {}\nexport { beta as gamma };\nclass Hidden {}\n', '.js'), ['alpha', 'gamma']);
  assert.deepEqual(parseExports('class Widget:\n    def render(self):\n        pass\n', '.py'), ['Widget', 'render']);
  assert.deepEqual(parseExports('type Order struct {}\nfunc Total(o Order) int { return 0 }\n', '.go'), ['Order', 'Total']);
  assert.deepEqual(parseExports('pub struct Order;\npub fn total() -> u32 { 0 }\nfn hidden() {}\n', '.rs'), ['Order', 'total']);
  assert.deepEqual(parseExports('module Shop\n  class Order\n    def total\n    end\n  end\nend\n', '.rb'), ['Shop', 'Order', 'total']);
  assert.deepEqual(parseExports('public class Order {\n    public int total() { return 0; }\n}\n', '.java'), ['Order', 'total']);
  assert.deepEqual(parseExports('public class Order {\n    fun total(): Int = 0\n}\n', '.kt'), ['Order', 'total']);
  assert.ok(parseExports('export function alpha() {}\n').includes('alpha'), 'the extension is optional');
});

test('repoBrief reports language names and test directories', async () => {
  const root = writeFiles(tempDir('brieflang'), {
    'package.json': JSON.stringify({ name: 'langs', scripts: { test: 'node --test' } }),
    'src/a.js': 'export function alpha() {}\n',
    'src/b.mjs': 'export function beta() {}\n',
    'src/c.py': 'def gamma():\n    pass\n',
    'tests/a.test.js': 'import test from "node:test";\n',
    'pkg/__tests__/b.js': 'export const b = 1;\n',
    'node_modules/dep/index.js': 'module.exports = 1;\n',
  });
  const brief = await repoBrief(null, fakeWorktree(root), 'sha-lang');
  assert.equal(brief.languages.js, 4, 'mjs counts as js');
  assert.equal(brief.languages.py, 1);
  assert.equal(brief.languages['.mjs'], undefined, 'extensions are not language keys');
  assert.deepEqual(brief.testDirs, ['pkg/__tests__', 'tests']);
  assert.ok(!brief.tree.some((f) => f.startsWith('node_modules/')), 'vendored files stay out');
  assert.deepEqual(testDirsFrom(['src/foo_test.go', 'a/b/c.js']), ['src']);
  fs.rmSync(root, { recursive: true, force: true });
});

test('repoKeyFor prefers the explicit key over the worktree path', () => {
  assert.equal(repoKeyFor({ path: '/tmp/wt' }, { repoKey: 'github.com/acme/app' }), 'github.com/acme/app');
  assert.equal(repoKeyFor({ repoPath: '/srv/app', path: '/tmp/wt' }), '/srv/app');
  assert.equal(repoKeyFor(null), 'repo');
});
