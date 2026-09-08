// kolm code — worktree drivers (spec section 4).
//
// The local driver is exercised against throwaway git repositories created
// under os.tmpdir(); nothing here touches the kolm repository, its index or the
// network. The fake driver is exercised in memory.

import test, { after } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

import { createDriver, DRIVER_KINDS } from '../src/ladder/worktree/index.js';
import { createLocalGitDriver } from '../src/ladder/worktree/local-git.js';
import { createFakeDriver, globToRegExp } from '../src/ladder/worktree/fake.js';

const gitProbe = spawnSync('git', ['--version'], { encoding: 'utf8' });
const gitMissing = gitProbe.error || gitProbe.status !== 0;
const skipWithoutGit = gitMissing
  ? 'git --version failed, so the local worktree driver cannot be exercised here'
  : false;

const scratchDirs = [];

after(async () => {
  for (const dir of scratchDirs) {
    await fs.rm(dir, { recursive: true, force: true, maxRetries: 5, retryDelay: 50 }).catch(() => {});
  }
});

async function scratch(prefix) {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), prefix));
  scratchDirs.push(dir);
  return await fs.realpath(dir);
}

function git(args, cwd) {
  const result = spawnSync('git', args, { cwd, encoding: 'utf8' });
  assert.equal(result.status, 0, `git ${args.join(' ')} failed: ${result.stderr}`);
  return result.stdout;
}

/** A fresh repository with one commit: README.md, src/a.js, src/nested/b.js, doomed.txt. */
async function makeRepo() {
  const dir = await scratch('kolm-code-repo-');
  git(['init', '-q', '-b', 'main'], dir);
  for (const [key, value] of [
    ['core.autocrlf', 'false'],
    ['commit.gpgsign', 'false'],
    ['user.name', 'kolm code test'],
    ['user.email', 'test@kolm.ai'],
  ]) {
    git(['config', key, value], dir);
  }
  await fs.mkdir(path.join(dir, 'src', 'nested'), { recursive: true });
  await fs.writeFile(path.join(dir, 'README.md'), 'hello\nworld\n', 'utf8');
  await fs.writeFile(path.join(dir, 'src', 'a.js'), 'export const a = 1;\n', 'utf8');
  await fs.writeFile(path.join(dir, 'src', 'nested', 'b.js'), 'export const b = 2;\n', 'utf8');
  await fs.writeFile(path.join(dir, 'doomed.txt'), 'delete me\n', 'utf8');
  // Explicit paths only; the repository under test is never staged wholesale.
  git(['add', 'README.md', 'src/a.js', 'src/nested/b.js', 'doomed.txt'], dir);
  git(['commit', '-q', '-m', 'base'], dir);
  return dir;
}

async function makeDriver() {
  const worktreesDir = await scratch('kolm-code-wt-');
  return { worktreesDir, driver: createLocalGitDriver({}, { worktreesDir }) };
}

async function exists(target) {
  try {
    await fs.stat(target);
    return true;
  } catch {
    return false;
  }
}

/* ------------------------------------------------------------- factory */

test('the driver factory builds every supported kind and refuses unknown ones', () => {
  assert.deepEqual([...DRIVER_KINDS], ['local', 'cloudflare', 'fake']);
  assert.equal(createDriver('fake', {}, { files: {} }).kind, 'fake');
  assert.equal(createDriver('local', {}, { worktreesDir: os.tmpdir() }).kind, 'local');
  assert.equal(createDriver('cloudflare', {}).kind, 'cloudflare');
  assert.throws(() => createDriver('sqlite'), (err) => err.code === 'bad-config');
});

/* -------------------------------------------------------- local driver */

test('the local driver creates a detached worktree at the base commit', { skip: skipWithoutGit }, async () => {
  const repoPath = await makeRepo();
  const { worktreesDir, driver } = await makeDriver();
  const wt = await driver.create({ repoPath, runId: 'run1', candidateId: 'cand1' });

  assert.equal(wt.kind, 'local');
  assert.equal(wt.id, 'cand1');
  assert.equal(wt.path, path.join(worktreesDir, 'run1', 'cand1'));
  assert.equal(wt.branch, 'kolm/code/run1/cand1');
  assert.equal(await wt.baseSha(), git(['rev-parse', 'HEAD'], repoPath).trim());
  assert.equal(await exists(path.join(wt.path, 'src', 'a.js')), true);
  assert.equal(git(['rev-parse', '--abbrev-ref', 'HEAD'], wt.path).trim(), 'HEAD',
    'the candidate worktree stays detached so branches remain free');
  await wt.destroy();
});

test('creating over an existing worktree is refused unless reuse is requested', { skip: skipWithoutGit }, async () => {
  const repoPath = await makeRepo();
  const { driver } = await makeDriver();
  const first = await driver.create({ repoPath, runId: 'run1', candidateId: 'cand1' });
  await assert.rejects(
    () => driver.create({ repoPath, runId: 'run1', candidateId: 'cand1' }),
    (err) => err.code === 'worktree-exists',
  );
  const reused = await driver.create({ repoPath, runId: 'run1', candidateId: 'cand1', reuse: true });
  assert.equal(reused.path, first.path);
  assert.equal(await reused.baseSha(), await first.baseSha());
  await reused.destroy();
});

test('a path outside a git repository is reported as such', { skip: skipWithoutGit }, async () => {
  const plain = await scratch('kolm-code-plain-');
  const { driver } = await makeDriver();
  await assert.rejects(
    () => driver.create({ repoPath: plain, runId: 'run1', candidateId: 'cand1' }),
    (err) => err.code === 'not-a-git-repo',
  );
});

test('an unresolvable base ref is reported before any worktree is created', { skip: skipWithoutGit }, async () => {
  const repoPath = await makeRepo();
  const { driver } = await makeDriver();
  await assert.rejects(
    () => driver.create({ repoPath, baseRef: 'no-such-ref', runId: 'run1', candidateId: 'cand1' }),
    (err) => err.code === 'bad-ref',
  );
});

test('reading writing and deleting files round trips inside the worktree', { skip: skipWithoutGit }, async () => {
  const repoPath = await makeRepo();
  const { driver } = await makeDriver();
  const wt = await driver.create({ repoPath, runId: 'run1', candidateId: 'cand1' });

  assert.equal(await wt.readFile('README.md'), 'hello\nworld\n');
  assert.equal(await wt.readFile('src/nested/b.js'), 'export const b = 2;\n');
  assert.equal(await wt.readFile('missing/file.txt'), null);

  await wt.writeFile('src/deep/new/file.txt', 'created\n');
  assert.equal(await wt.readFile('src/deep/new/file.txt'), 'created\n');
  assert.equal(
    await fs.readFile(path.join(wt.path, 'src', 'deep', 'new', 'file.txt'), 'utf8'),
    'created\n',
  );

  await wt.deleteFile('doomed.txt');
  assert.equal(await wt.readFile('doomed.txt'), null);
  await wt.deleteFile('doomed.txt');
  assert.equal(await exists(path.join(wt.path, 'doomed.txt')), false);
  await wt.destroy();
});

test('paths that escape the worktree are refused by every file operation', { skip: skipWithoutGit }, async () => {
  const repoPath = await makeRepo();
  const { driver } = await makeDriver();
  const wt = await driver.create({ repoPath, runId: 'run1', candidateId: 'cand1' });
  const escapes = ['../escape.txt', 'src/../../escape.txt', path.join(os.tmpdir(), 'escape.txt'), '/etc/passwd'];
  for (const bad of escapes) {
    await assert.rejects(() => wt.readFile(bad), (err) => err.code === 'path-escape', `readFile ${bad}`);
    await assert.rejects(() => wt.writeFile(bad, 'x'), (err) => err.code === 'path-escape', `writeFile ${bad}`);
    await assert.rejects(() => wt.deleteFile(bad), (err) => err.code === 'path-escape', `deleteFile ${bad}`);
  }
  await assert.rejects(
    () => wt.exec('node -e "0"', { cwd: '../..' }),
    (err) => err.code === 'path-escape',
  );
  assert.equal(await exists(path.join(path.dirname(wt.path), 'escape.txt')), false);
  await wt.destroy();
});

test('listing files covers tracked and untracked entries and honours a glob', { skip: skipWithoutGit }, async () => {
  const repoPath = await makeRepo();
  const { driver } = await makeDriver();
  const wt = await driver.create({ repoPath, runId: 'run1', candidateId: 'cand1' });
  await wt.writeFile('src/fresh.js', 'export const fresh = 3;\n');
  await wt.deleteFile('doomed.txt');

  const all = await wt.listFiles();
  assert.deepEqual(all, ['README.md', 'src/a.js', 'src/fresh.js', 'src/nested/b.js']);
  assert.equal(all.includes('doomed.txt'), false, 'a file deleted from the working tree is not listed');

  const scoped = await wt.listFiles({ glob: 'src/*.js' });
  assert.ok(scoped.includes('src/a.js') && scoped.includes('src/fresh.js'));
  assert.equal(scoped.includes('README.md'), false);
  await wt.destroy();
});

test('searching returns path line and text triples and an empty list for no match', { skip: skipWithoutGit }, async () => {
  const repoPath = await makeRepo();
  const { driver } = await makeDriver();
  const wt = await driver.create({ repoPath, runId: 'run1', candidateId: 'cand1' });
  await wt.writeFile('src/needle.js', 'const first = 1;\nconst needle = 2;\n');

  const hits = await wt.search('needle');
  const inFile = hits.filter((hit) => hit.path === 'src/needle.js');
  assert.equal(inFile.length, 1);
  assert.equal(inFile[0].line, 2);
  assert.match(inFile[0].text, /const needle = 2;/);
  assert.deepEqual(await wt.search('definitely-absent-token'), []);
  await wt.destroy();
});

test('the diff covers a tracked modification an untracked file and a deletion', { skip: skipWithoutGit }, async () => {
  const repoPath = await makeRepo();
  const { driver } = await makeDriver();
  const wt = await driver.create({ repoPath, runId: 'run1', candidateId: 'cand1' });
  await wt.writeFile('README.md', 'hello\nchanged\n');
  await wt.writeFile('src/fresh.js', 'export const fresh = 3;\n');
  await wt.deleteFile('doomed.txt');

  const diff = await wt.diff();
  assert.match(diff, /diff --git a\/README\.md b\/README\.md/);
  assert.match(diff, /-world\n\+changed/);
  assert.match(diff, /diff --git a\/src\/fresh\.js b\/src\/fresh\.js/);
  assert.match(diff, /new file mode 100644/);
  assert.match(diff, /diff --git a\/doomed\.txt b\/doomed\.txt/);
  assert.match(diff, /deleted file mode 100644/);
  assert.match(diff, /@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@/);

  const changed = await wt.changedFiles();
  const byPath = new Map(changed.map((row) => [row.path, row]));
  assert.deepEqual(byPath.get('README.md'), { path: 'README.md', status: 'M', added: 1, deleted: 1 });
  assert.deepEqual(byPath.get('src/fresh.js'), { path: 'src/fresh.js', status: 'A', added: 1, deleted: 0 });
  assert.deepEqual(byPath.get('doomed.txt'), { path: 'doomed.txt', status: 'D', added: 0, deleted: 1 });
  await wt.destroy();
});

test('a diff produced by one worktree applies cleanly to a second one', { skip: skipWithoutGit }, async () => {
  const repoPath = await makeRepo();
  const { driver } = await makeDriver();
  const source = await driver.create({ repoPath, runId: 'run1', candidateId: 'cand1' });
  await source.writeFile('README.md', 'hello\nchanged\n');
  await source.writeFile('src/fresh.js', 'export const fresh = 3;\n');
  await source.deleteFile('doomed.txt');
  const diff = await source.diff();

  const target = await driver.create({ repoPath, runId: 'run1', candidateId: 'cand2' });
  const applied = await target.applyPatch(diff);
  assert.deepEqual(applied, { ok: true });
  assert.equal(await target.readFile('README.md'), 'hello\nchanged\n');
  assert.equal(await target.readFile('src/fresh.js'), 'export const fresh = 3;\n');
  assert.equal(await target.readFile('doomed.txt'), null);
  assert.deepEqual(await target.changedFiles(), await source.changedFiles());

  const rejected = await target.applyPatch('diff --git a/x b/x\n@@ -1 +1 @@\n-nope\n+nope\n');
  assert.equal(rejected.ok, false);
  assert.equal(typeof rejected.error, 'string');
  assert.equal((await target.applyPatch('')).ok, false);

  await source.destroy();
  await target.destroy();
});

test('committing points the candidate branch at the new commit in the main repository', { skip: skipWithoutGit }, async () => {
  const repoPath = await makeRepo();
  const { driver } = await makeDriver();
  const wt = await driver.create({ repoPath, runId: 'run1', candidateId: 'cand1' });
  await wt.writeFile('src/fresh.js', 'export const fresh = 3;\n');
  await wt.deleteFile('doomed.txt');

  const { sha, branch } = await wt.commit('candidate work');
  assert.equal(branch, 'kolm/code/run1/cand1');
  assert.match(sha, /^[0-9a-f]{40}$/);
  assert.equal(git(['rev-parse', '--verify', branch], repoPath).trim(), sha);
  assert.equal(git(['rev-parse', 'HEAD'], wt.path).trim(), sha);
  assert.equal(git(['rev-parse', '--abbrev-ref', 'HEAD'], wt.path).trim(), 'HEAD',
    'the worktree stays detached so the branch can be checked out elsewhere');
  assert.match(git(['show', '--stat', '--format=%s', sha], repoPath), /candidate work/);
  assert.equal(git(['status', '--porcelain'], wt.path).trim(), '', 'nothing is left uncommitted');
  assert.deepEqual(
    (await wt.changedFiles()).map((row) => row.path),
    ['doomed.txt', 'src/fresh.js'],
    'the diff stays relative to the base commit after committing',
  );

  // A resumed run commits a worktree that was already committed before the
  // interruption: a clean tree is not an error, HEAD is simply re-reported and
  // the branch is re-pointed at it.
  const again = await wt.commit('candidate work');
  assert.deepEqual(again, { sha, branch });
  assert.equal(git(['rev-parse', '--verify', branch], repoPath).trim(), sha);
  assert.equal(git(['rev-list', '--count', sha], repoPath).trim(), '2', 'no empty commit was added');
  await wt.destroy();
});

test('committing a worktree with no changes reports the base commit instead of failing', { skip: skipWithoutGit }, async () => {
  const repoPath = await makeRepo();
  const { driver } = await makeDriver();
  const base = git(['rev-parse', 'HEAD'], repoPath).trim();
  const wt = await driver.create({ repoPath, runId: 'run1', candidateId: 'cand1' });

  const { sha, branch } = await wt.commit('nothing to do');
  assert.equal(sha, base);
  assert.equal(branch, 'kolm/code/run1/cand1');
  assert.equal(git(['rev-parse', '--verify', branch], repoPath).trim(), base);
  await wt.destroy();
});

test('exec reports a non-zero exit code without throwing', { skip: skipWithoutGit }, async () => {
  const repoPath = await makeRepo();
  const { driver } = await makeDriver();
  const wt = await driver.create({ repoPath, runId: 'run1', candidateId: 'cand1' });
  const result = await wt.exec('node -e "process.stdout.write(\'out\');process.stderr.write(\'err\');process.exit(3)"');
  assert.equal(result.exitCode, 3);
  assert.match(result.stdout, /out/);
  assert.match(result.stderr, /err/);
  assert.equal(result.timedOut, false);
  assert.equal(typeof result.durationMs, 'number');
  await wt.destroy();
});

test('exec runs in the worktree and accepts a relative working directory', { skip: skipWithoutGit }, async () => {
  const repoPath = await makeRepo();
  const { driver } = await makeDriver();
  const wt = await driver.create({ repoPath, runId: 'run1', candidateId: 'cand1' });
  const here = await wt.exec('node -e "process.stdout.write(process.cwd())"');
  assert.equal(await fs.realpath(here.stdout.trim()), await fs.realpath(wt.path));
  const nested = await wt.exec('node -e "process.stdout.write(process.cwd())"', { cwd: 'src/nested' });
  assert.equal(await fs.realpath(nested.stdout.trim()), await fs.realpath(path.join(wt.path, 'src', 'nested')));
  await wt.destroy();
});

test('exec kills a command that exceeds its timeout and reports it', { skip: skipWithoutGit }, async () => {
  const repoPath = await makeRepo();
  const { driver } = await makeDriver();
  const wt = await driver.create({ repoPath, runId: 'run1', candidateId: 'cand1' });
  const started = Date.now();
  const result = await wt.exec('node -e "setTimeout(()=>{},10000)"', { timeoutMs: 500 });
  const elapsed = Date.now() - started;
  assert.equal(result.timedOut, true);
  assert.notEqual(result.exitCode, 0);
  assert.ok(elapsed < 9000, `the timed-out command must return promptly, took ${elapsed}ms`);
  await wt.destroy();
});

test('exec truncates output beyond the configured byte cap', { skip: skipWithoutGit }, async () => {
  const repoPath = await makeRepo();
  const { driver } = await makeDriver();
  const wt = await driver.create({ repoPath, runId: 'run1', candidateId: 'cand1' });
  const result = await wt.exec('node -e "process.stdout.write(\'x\'.repeat(5000))"', { maxOutputBytes: 100 });
  assert.equal(result.truncated, true);
  assert.ok(result.stdout.length <= 100, `stdout was ${result.stdout.length} bytes`);
  await wt.destroy();
});

test('destroying a worktree removes its directory and prunes the repository', { skip: skipWithoutGit }, async () => {
  const repoPath = await makeRepo();
  const { driver } = await makeDriver();
  const wt = await driver.create({ repoPath, runId: 'run1', candidateId: 'cand1' });
  assert.match(git(['worktree', 'list', '--porcelain'], repoPath), /worktree /);
  await wt.destroy();
  assert.equal(await exists(wt.path), false);
  const listed = git(['worktree', 'list', '--porcelain'], repoPath);
  assert.equal(listed.includes(path.basename(wt.path)), false, 'the removed worktree is pruned from the repository');
  await wt.destroy();
});

test('destroying a run removes every worktree that belongs to it', { skip: skipWithoutGit }, async () => {
  const repoPath = await makeRepo();
  const { worktreesDir, driver } = await makeDriver();
  const first = await driver.create({ repoPath, runId: 'run7', candidateId: 'cand1' });
  const second = await driver.create({ repoPath, runId: 'run7', candidateId: 'cand2' });
  const other = await driver.create({ repoPath, runId: 'run8', candidateId: 'cand1' });

  const { removed } = await driver.destroyRun('run7');
  assert.ok(removed >= 2, `expected at least two removals, got ${removed}`);
  assert.equal(await exists(first.path), false);
  assert.equal(await exists(second.path), false);
  assert.equal(await exists(path.join(worktreesDir, 'run7')), false);
  assert.equal(await exists(other.path), true, 'another run is untouched');
  await other.destroy();
});

test('open re-attaches to an existing worktree from a fresh driver instance', { skip: skipWithoutGit }, async () => {
  const repoPath = await makeRepo();
  const { worktreesDir, driver } = await makeDriver();
  const created = await driver.create({ repoPath, runId: 'run1', candidateId: 'cand1' });
  await created.writeFile('resume.txt', 'still here\n');

  const resumed = createLocalGitDriver({ worktreesDir });
  const wt = await resumed.open({ repoPath, runId: 'run1', candidateId: 'cand1' });
  assert.equal(wt.path, created.path);
  assert.equal(await wt.baseSha(), await created.baseSha());
  assert.equal(await wt.readFile('resume.txt'), 'still here\n');

  await assert.rejects(
    () => resumed.open({ repoPath, runId: 'run1', candidateId: 'nothere' }),
    (err) => typeof err.code === 'string',
  );
  await wt.destroy();
});

test('the local driver rejects identifiers that are not simple path segments', { skip: skipWithoutGit }, async () => {
  const repoPath = await makeRepo();
  const { driver } = await makeDriver();
  for (const runId of ['../escape', 'has/slash', '', '.hidden']) {
    await assert.rejects(
      () => driver.create({ repoPath, runId, candidateId: 'cand1' }),
      (err) => err.code === 'bad-config',
      `runId ${JSON.stringify(runId)}`,
    );
  }
});

/* --------------------------------------------------------- fake driver */

const FAKE_FILES = {
  'README.md': 'hello\nworld\n',
  'src/a.js': 'export const a = 1;\n',
  'src/nested/b.js': 'export const b = 2;\n',
  'doomed.txt': 'delete me\n',
};

test('the fake driver gives every worktree its own copy of the base files', async () => {
  const driver = createFakeDriver({ files: FAKE_FILES });
  const one = await driver.create({ runId: 'run1', candidateId: 'cand1' });
  const two = await driver.create({ runId: 'run1', candidateId: 'cand2' });
  await one.writeFile('src/a.js', 'export const a = 99;\n');
  assert.equal(await one.readFile('src/a.js'), 'export const a = 99;\n');
  assert.equal(await two.readFile('src/a.js'), 'export const a = 1;\n');
  assert.equal(await two.readFile('nope.js'), null);
  assert.equal(await one.baseSha(), driver.baseSha);
  assert.deepEqual(driver.created.map((row) => row.candidateId), ['cand1', 'cand2']);
});

test('the fake driver refuses paths that would escape the worktree', async () => {
  const driver = createFakeDriver({ files: FAKE_FILES });
  const wt = await driver.create({ runId: 'run1', candidateId: 'cand1' });
  for (const bad of ['../escape', '/abs/path', 'a/../../b', 'C:/windows', '']) {
    await assert.rejects(() => wt.writeFile(bad, 'x'), (err) => err.code === 'path-escape', `writeFile ${bad}`);
  }
});

test('the fake exec table answers by exact command and fails loud otherwise', async () => {
  const driver = createFakeDriver({
    files: FAKE_FILES,
    exec: {
      'npm test --silent': { stdout: 'ok\n', exitCode: 0 },
      'node build.js': (cmd, wt) => ({ exitCode: wt.readFileSync('src/a.js') ? 0 : 1, stdout: cmd }),
    },
  });
  const wt = await driver.create({ runId: 'run1', candidateId: 'cand1' });

  const scripted = await wt.exec('npm test --silent');
  assert.deepEqual(scripted, {
    stdout: 'ok\n', stderr: '', exitCode: 0, durationMs: 0, timedOut: false, truncated: false,
  });

  const computed = await wt.exec('node build.js', { cwd: 'src', timeoutMs: 1000 });
  assert.equal(computed.exitCode, 0);
  assert.equal(computed.stdout, 'node build.js');

  const missing = await wt.exec('npm run nope');
  assert.equal(missing.exitCode, 127);
  assert.match(missing.stderr, /fake: no script for npm run nope/);

  assert.deepEqual(wt.commands.map((row) => row.cmd), ['npm test --silent', 'node build.js', 'npm run nope']);
  assert.equal(driver.commands.length, 3);
  assert.equal(wt.commands[1].cwd, 'src');
  assert.equal(wt.commands[1].timeoutMs, 1000);
  await assert.rejects(() => wt.exec('  '), (err) => err.code === 'bad-config');
});

test('the fake diff is a unified diff a parser can read', async () => {
  const driver = createFakeDriver({ files: FAKE_FILES });
  const wt = await driver.create({ runId: 'run1', candidateId: 'cand1' });
  await wt.writeFile('README.md', 'hello\nchanged\n');
  await wt.writeFile('src/fresh.js', 'export const fresh = 3;\n');
  await wt.deleteFile('doomed.txt');

  const diff = await wt.diff();
  assert.match(diff, /^diff --git a\/README\.md b\/README\.md$/m);
  assert.match(diff, /^--- a\/README\.md$/m);
  assert.match(diff, /^\+\+\+ b\/README\.md$/m);
  assert.match(diff, /^@@ -1,2 \+1,2 @@$/m);
  assert.match(diff, /^new file mode 100644$/m);
  assert.match(diff, /^--- \/dev\/null$/m);
  assert.match(diff, /^deleted file mode 100644$/m);
  assert.match(diff, /^\+\+\+ \/dev\/null$/m);
  assert.match(diff, /^index [0-9a-f]{7}\.\.[0-9a-f]{7}( 100644)?$/m);
  assert.equal(diff.split('\n').filter((line) => line.startsWith('diff --git ')).length, 3);

  for (const line of diff.split('\n')) {
    const header = /^@@ -(\d+),(\d+) \+(\d+),(\d+) @@$/.exec(line);
    if (!header) continue;
    const [, oldStart, oldCount, newStart, newCount] = header.map(Number);
    assert.ok(oldCount === 0 ? oldStart === 0 : oldStart === 1, 'old hunk start is consistent with its count');
    assert.ok(newCount === 0 ? newStart === 0 : newStart === 1, 'new hunk start is consistent with its count');
  }

  assert.deepEqual(await wt.changedFiles(), [
    { path: 'README.md', status: 'M', added: 2, deleted: 2 },
    { path: 'doomed.txt', status: 'D', added: 0, deleted: 1 },
    { path: 'src/fresh.js', status: 'A', added: 1, deleted: 0 },
  ]);
});

test('a fake diff applies onto a second fake worktree and reproduces the tree', async () => {
  const driver = createFakeDriver({ files: FAKE_FILES });
  const source = await driver.create({ runId: 'run1', candidateId: 'cand1' });
  await source.writeFile('README.md', 'hello\nchanged\n');
  await source.writeFile('src/fresh.js', 'export const fresh = 3;\n');
  await source.deleteFile('doomed.txt');

  const target = await driver.create({ runId: 'run1', candidateId: 'cand2' });
  assert.deepEqual(await target.applyPatch(await source.diff()), { ok: true });
  assert.deepEqual(await target.listFiles(), await source.listFiles());
  for (const file of await source.listFiles()) {
    assert.equal(await target.readFile(file), await source.readFile(file), file);
  }
  assert.deepEqual(await target.changedFiles(), await source.changedFiles());

  const broken = await target.applyPatch('diff --git a/src/a.js b/src/a.js\n--- a/src/a.js\n+++ b/src/a.js\n@@ -1,1 +1,1 @@\n-not the real line\n+replacement\n');
  assert.equal(broken.ok, false);
  assert.match(broken.error, /src\/a\.js/);
  assert.equal((await target.applyPatch('   ')).ok, false);
});

test('fake globs support single star double star and question mark', async () => {
  const driver = createFakeDriver({
    files: { ...FAKE_FILES, 'src/c1.js': 'c\n', 'src/deep/inner/d.js': 'd\n' },
  });
  const wt = await driver.create({ runId: 'run1', candidateId: 'cand1' });
  assert.deepEqual(await wt.listFiles({ glob: 'src/*.js' }), ['src/a.js', 'src/c1.js']);
  assert.deepEqual(await wt.listFiles({ glob: 'src/**/*.js' }),
    ['src/a.js', 'src/c1.js', 'src/deep/inner/d.js', 'src/nested/b.js']);
  assert.deepEqual(await wt.listFiles({ glob: 'src/c?.js' }), ['src/c1.js']);
  assert.deepEqual(await wt.listFiles({ glob: '*.md' }), ['README.md']);
  assert.equal((await wt.listFiles({ limit: 2 })).length, 2);
  assert.equal(globToRegExp('src/*.js').test('src/nested/b.js'), false);
  assert.equal(globToRegExp('**/*.js').test('src/nested/b.js'), true);
});

test('fake search matches a regular expression line by line', async () => {
  const driver = createFakeDriver({
    files: { 'src/a.js': 'const needle = 1;\nconst other = 2;\n', 'notes.md': 'needle here\n' },
  });
  const wt = await driver.create({ runId: 'run1', candidateId: 'cand1' });
  assert.deepEqual(await wt.search('needle'), [
    { path: 'notes.md', line: 1, text: 'needle here' },
    { path: 'src/a.js', line: 1, text: 'const needle = 1;' },
  ]);
  assert.deepEqual(await wt.search('needle', { glob: 'src/*.js' }), [
    { path: 'src/a.js', line: 1, text: 'const needle = 1;' },
  ]);
  assert.deepEqual(await wt.search('nothing-here'), []);
  await assert.rejects(() => wt.search('('), (err) => err.code === 'bad-config');
});

test('fake commits are deterministic and destruction is recorded', async () => {
  const driver = createFakeDriver({ files: FAKE_FILES });
  const wt = await driver.create({ runId: 'run1', candidateId: 'cand1' });
  const first = await wt.commit('one');
  const second = await wt.commit('two');
  assert.match(first.sha, /^[0-9a-f]{40}$/);
  assert.notEqual(first.sha, second.sha);
  assert.equal(first.branch, 'kolm/code/run1/cand1');

  const twin = createFakeDriver({ files: FAKE_FILES });
  const twinWt = await twin.create({ runId: 'run1', candidateId: 'cand1' });
  assert.equal((await twinWt.commit('one')).sha, first.sha, 'the same run and candidate commit to the same sha');

  await wt.destroy();
  assert.deepEqual(driver.destroyed, [{ runId: 'run1', candidateId: 'cand1', id: 'cand1' }]);

  const other = createFakeDriver({ files: FAKE_FILES });
  await other.create({ runId: 'run2', candidateId: 'cand1' });
  await other.create({ runId: 'run2', candidateId: 'cand2' });
  await other.create({ runId: 'run3', candidateId: 'cand1' });
  assert.deepEqual(await other.destroyRun('run2'), { removed: 2 });
  assert.equal(other.worktrees.size, 1);
});

test('the fake driver reopens a worktree it already created', async () => {
  const driver = createFakeDriver({ files: FAKE_FILES });
  const wt = await driver.create({ runId: 'run1', candidateId: 'cand1' });
  await wt.writeFile('src/a.js', 'export const a = 42;\n');
  const reopened = await driver.open({ runId: 'run1', candidateId: 'cand1' });
  assert.equal(await reopened.readFile('src/a.js'), 'export const a = 42;\n');
  const fresh = await driver.open({ runId: 'run1', candidateId: 'cand9' });
  assert.equal(await fresh.readFile('src/a.js'), 'export const a = 1;\n');
});
