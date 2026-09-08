// Owner C — unified diff parsing used by L0 and the receipt writer.
import test from 'node:test';
import assert from 'node:assert/strict';
import { parseUnifiedDiff, diffStats } from '../src/ladder/diff-parse.js';

const MULTI_FILE = [
  'diff --git a/src/added.js b/src/added.js',
  'new file mode 100644',
  'index 0000000..1c2d3e4',
  '--- /dev/null',
  '+++ b/src/added.js',
  '@@ -0,0 +1,3 @@',
  '+export const one = 1;',
  '+export const two = 2;',
  '+export const three = 3;',
  'diff --git a/src/modified.js b/src/modified.js',
  'index aaaaaaa..bbbbbbb 100644',
  '--- a/src/modified.js',
  '+++ b/src/modified.js',
  '@@ -10,4 +10,5 @@ function keep() {',
  ' const untouched = true;',
  '-const removed = 1;',
  '+const addedA = 1;',
  '+const addedB = 2;',
  ' const alsoUntouched = true;',
  ' return untouched;',
  'diff --git a/src/gone.js b/src/gone.js',
  'deleted file mode 100644',
  'index ccccccc..0000000',
  '--- a/src/gone.js',
  '+++ /dev/null',
  '@@ -1,2 +0,0 @@',
  '-const a = 1;',
  '-const b = 2;',
  'diff --git a/old/name.js b/new/name.js',
  'similarity index 92%',
  'rename from old/name.js',
  'rename to new/name.js',
  '--- a/old/name.js',
  '+++ b/new/name.js',
  '@@ -1,1 +1,1 @@',
  '-const name = "old";',
  '+const name = "new";',
  'diff --git a/assets/logo.png b/assets/logo.png',
  'index 1111111..2222222 100644',
  'GIT binary patch',
  'literal 512',
  '',
].join('\n');

test('parseUnifiedDiff reads add modify delete rename and binary files', () => {
  const files = parseUnifiedDiff(MULTI_FILE);
  assert.equal(files.length, 5);

  const added = files[0];
  assert.equal(added.path, 'src/added.js');
  assert.equal(added.status, 'A');
  assert.equal(added.oldPath, null);
  assert.equal(added.added, 3);
  assert.equal(added.deleted, 0);
  assert.equal(added.binary, false);

  const modified = files[1];
  assert.equal(modified.path, 'src/modified.js');
  assert.equal(modified.status, 'M');
  assert.equal(modified.oldPath, 'src/modified.js');
  assert.equal(modified.added, 2);
  assert.equal(modified.deleted, 1);

  const deleted = files[2];
  assert.equal(deleted.path, 'src/gone.js');
  assert.equal(deleted.status, 'D');
  assert.equal(deleted.deleted, 2);
  assert.equal(deleted.added, 0);

  const renamed = files[3];
  assert.equal(renamed.status, 'R');
  assert.equal(renamed.oldPath, 'old/name.js');
  assert.equal(renamed.path, 'new/name.js');
  assert.equal(renamed.added, 1);
  assert.equal(renamed.deleted, 1);

  const binary = files[4];
  assert.equal(binary.path, 'assets/logo.png');
  assert.equal(binary.binary, true);
  assert.equal(binary.hunks.length, 0);
});

test('parseUnifiedDiff records hunk headers and typed lines', () => {
  const files = parseUnifiedDiff(MULTI_FILE);
  const hunk = files[1].hunks[0];
  assert.equal(hunk.oldStart, 10);
  assert.equal(hunk.oldLines, 4);
  assert.equal(hunk.newStart, 10);
  assert.equal(hunk.newLines, 5);
  assert.equal(hunk.lines.length, 6);
  assert.deepEqual(hunk.lines[0], { type: ' ', text: 'const untouched = true;' });
  assert.deepEqual(hunk.lines[1], { type: '-', text: 'const removed = 1;' });
  assert.deepEqual(hunk.lines[2], { type: '+', text: 'const addedA = 1;' });
});

test('diffStats totals files churn and binary paths', () => {
  const stats = diffStats(MULTI_FILE);
  assert.equal(stats.files, 5);
  assert.equal(stats.added, 3 + 2 + 0 + 1);
  assert.equal(stats.deleted, 0 + 1 + 2 + 1);
  assert.equal(stats.patchLines, stats.added + stats.deleted);
  assert.deepEqual(stats.binaries, ['assets/logo.png']);
});

test('diffStats on an empty or whitespace patch reports nothing', () => {
  for (const empty of ['', '   ', null, undefined]) {
    const stats = diffStats(empty);
    assert.equal(stats.files, 0);
    assert.equal(stats.added, 0);
    assert.equal(stats.deleted, 0);
    assert.equal(stats.patchLines, 0);
    assert.deepEqual(stats.binaries, []);
  }
});

test('parseUnifiedDiff never mistakes hunk body text for a file header', () => {
  const tricky = [
    'diff --git a/doc.md b/doc.md',
    '--- a/doc.md',
    '+++ b/doc.md',
    '@@ -1,4 +1,4 @@',
    ' intro',
    '--- a/not-a-header.js',
    '+++ b/not-a-header.js',
    '-diff --git a/fake b/fake',
    '+@@ -1,1 +1,1 @@',
    ' outro',
    ' tail',
  ].join('\n');
  const files = parseUnifiedDiff(tricky);
  assert.equal(files.length, 1);
  assert.equal(files[0].path, 'doc.md');
  assert.equal(files[0].hunks.length, 1);
  assert.equal(files[0].added, 2);
  assert.equal(files[0].deleted, 2);
  const texts = files[0].hunks[0].lines.map((l) => l.type + l.text);
  assert.ok(texts.includes('-diff --git a/fake b/fake'));
  assert.ok(texts.includes('+@@ -1,1 +1,1 @@'));
});

test('parseUnifiedDiff handles plain diff minus u output without git headers', () => {
  const plain = [
    '--- old/file.txt\t2024-01-01',
    '+++ new/file.txt\t2024-01-02',
    '@@ -1 +1 @@',
    '-before',
    '+after',
  ].join('\n');
  const files = parseUnifiedDiff(plain);
  assert.equal(files.length, 1);
  assert.equal(files[0].added, 1);
  assert.equal(files[0].deleted, 1);
  assert.equal(files[0].hunks[0].oldLines, 1);
  assert.equal(files[0].hunks[0].newLines, 1);
});

test('parseUnifiedDiff tolerates the no newline marker and CRLF input', () => {
  const crlf = [
    'diff --git a/a.txt b/a.txt',
    '--- a/a.txt',
    '+++ b/a.txt',
    '@@ -1,1 +1,1 @@',
    '-one',
    '\\ No newline at end of file',
    '+two',
    '\\ No newline at end of file',
  ].join('\r\n');
  const files = parseUnifiedDiff(crlf);
  assert.equal(files.length, 1);
  assert.equal(files[0].added, 1);
  assert.equal(files[0].deleted, 1);
});

test('parseUnifiedDiff reads the Binary files differ marker form', () => {
  const bin = [
    'diff --git a/img.gif b/img.gif',
    'index 3333333..4444444 100644',
    'Binary files a/img.gif and b/img.gif differ',
  ].join('\n');
  const files = parseUnifiedDiff(bin);
  assert.equal(files.length, 1);
  assert.equal(files[0].binary, true);
  assert.equal(files[0].path, 'img.gif');
  assert.deepEqual(diffStats(bin).binaries, ['img.gif']);
});
