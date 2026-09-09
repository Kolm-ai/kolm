// Owner CTX - lessons carried across runs in one repo (SOTA spec 1.5).
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {
  lessonsPath,
  loadLessons,
  saveLessons,
  mergeLessons,
  lessonsToPrompt,
  extractLessonCandidates,
  repoKeySlug,
  LESSONS_CAP,
  LESSONS_BYTE_CAP,
  LESSON_TEXT_CAP,
  LESSON_CANDIDATE_CAP,
} from '../src/ladder/lessons.js';

function tempDir(label) {
  return fs.mkdtempSync(path.join(os.tmpdir(), `kolm-lessons-${label}-`));
}

/** Only the two fields lessons.js reads off the config object. */
function fakeConfig(dataDir, lessons) {
  return lessons === undefined ? { dataDir } : { dataDir, lessons };
}

const REPO = 'C:/Users/dev/Desktop/app';

// ------------------------------------------------------------------- paths

test('lessonsPath lives under dataDir/code/lessons and separates repos', () => {
  const dir = tempDir('path');
  const file = lessonsPath(fakeConfig(dir), REPO);
  assert.equal(path.dirname(file), path.join(dir, 'code', 'lessons'));
  assert.ok(file.endsWith('.json'));

  // Two repos whose slugs collide still get different files: the digest of the
  // raw key is what keeps them apart.
  const a = lessonsPath(fakeConfig(dir), '/srv/app');
  const b = lessonsPath(fakeConfig(dir), '/srv/app/');
  assert.notEqual(a, b);
  assert.equal(lessonsPath(fakeConfig(dir), REPO), file, 'the same key always maps to the same file');
  assert.match(repoKeySlug(REPO), /^[A-Za-z0-9._-]+$/, 'the slug is filename safe');
  assert.equal(repoKeySlug('///').startsWith('repo-'), true, 'a key with nothing usable still gets a name');
  fs.rmSync(dir, { recursive: true, force: true });
});

// ------------------------------------------------------------- load / save

test('saveLessons then loadLessons round-trips the document', () => {
  const dir = tempDir('roundtrip');
  const config = fakeConfig(dir);
  saveLessons(config, REPO, [
    { text: 'Stage l1 skips in this repo: no lint command detected.', runId: 'run-1', ts: '2026-09-01T00:00:00.000Z', hits: 2 },
    { text: 'The run tool refuses "npm install" in this repo.', runId: 'run-1', ts: '2026-09-01T00:00:00.000Z', hits: 1 },
  ]);
  const doc = loadLessons(config, REPO);
  assert.equal(doc.version, 1);
  assert.equal(doc.repoKey, REPO);
  assert.equal(doc.lessons.length, 2);
  assert.equal(doc.lessons[0].text, 'Stage l1 skips in this repo: no lint command detected.');
  assert.equal(doc.lessons[0].hits, 2);
  assert.equal(typeof doc.lessons[0].id, 'string');

  const file = lessonsPath(config, REPO);
  assert.equal(fs.existsSync(file), true);
  assert.deepEqual(fs.readdirSync(path.dirname(file)).filter((f) => f.includes('.tmp.')), [], 'no temp file is left behind');
  if (process.platform !== 'win32') {
    assert.equal(fs.statSync(file).mode & 0o777, 0o600, 'the file is owner-only');
  }
  fs.rmSync(dir, { recursive: true, force: true });
});

test('loadLessons reads a missing or corrupt file as no lessons', () => {
  const dir = tempDir('corrupt');
  const config = fakeConfig(dir);
  assert.deepEqual(loadLessons(config, REPO).lessons, [], 'a missing file is not an error');

  const file = lessonsPath(config, REPO);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, '{ this is not json');
  assert.deepEqual(loadLessons(config, REPO).lessons, []);

  fs.writeFileSync(file, JSON.stringify({ version: 1, repoKey: REPO, lessons: [{ nope: true }, 'a bare string lesson', { text: '   ' }] }));
  const doc = loadLessons(config, REPO);
  assert.deepEqual(doc.lessons.map((l) => l.text), ['a bare string lesson'], 'malformed entries are dropped');
  fs.rmSync(dir, { recursive: true, force: true });
});

test('config.lessons false disables both halves', () => {
  const dir = tempDir('off');
  const off = fakeConfig(dir, false);
  saveLessons(off, REPO, [{ text: 'never written' }]);
  assert.equal(fs.existsSync(lessonsPath(off, REPO)), false, 'nothing is written when lessons are off');
  assert.deepEqual(loadLessons(off, REPO).lessons, []);

  // With lessons on, the same call writes.
  const on = fakeConfig(dir, true);
  saveLessons(on, REPO, [{ text: 'written' }]);
  assert.equal(loadLessons(on, REPO).lessons.length, 1);
  assert.deepEqual(loadLessons(off, REPO).lessons, [], 'an off config does not read an existing file either');
  fs.rmSync(dir, { recursive: true, force: true });
});

test('saveLessons swallows an unwritable data directory', () => {
  const dir = tempDir('unwritable');
  // A file where the directory should be: mkdir fails, and so must the write.
  const blocked = path.join(dir, 'block');
  fs.writeFileSync(blocked, 'not a directory');
  const config = fakeConfig(blocked);
  assert.doesNotThrow(() => saveLessons(config, REPO, [{ text: 'lost, but not fatal' }]));
  assert.deepEqual(loadLessons(config, REPO).lessons, []);
  fs.rmSync(dir, { recursive: true, force: true });
});

// ----------------------------------------------------------------- merging

test('mergeLessons dedupes on the case-insensitive prefix and counts hits', () => {
  const existing = [{ text: 'Stage l1 skips in this repo: no lint command detected.', ts: '2026-09-01T00:00:00.000Z', hits: 1 }];
  const merged = mergeLessons(existing, ['STAGE L1 skips in this repo: no lint command detected'], { runId: 'run-2', ts: '2026-09-02T00:00:00.000Z' });
  assert.equal(merged.length, 1, 'the same lesson twice is one lesson');
  assert.equal(merged[0].hits, 2);
  assert.equal(merged[0].runId, 'run-2');
  assert.equal(merged[0].ts, '2026-09-02T00:00:00.000Z');
});

test('mergeLessons keeps the newest and caps the list', () => {
  const older = [];
  for (let i = 0; i < LESSONS_CAP; i += 1) {
    older.push({ text: `Old lesson number ${i} about this repo.`, ts: '2026-08-01T00:00:00.000Z', hits: 1 });
  }
  const merged = mergeLessons(older, ['A brand new lesson about this repo.'], { ts: '2026-09-09T00:00:00.000Z' });
  assert.equal(merged.length, LESSONS_CAP);
  assert.equal(merged[0].text, 'A brand new lesson about this repo.', 'newest first');
  assert.equal(merged.some((l) => l.text === 'Old lesson number 0 about this repo.'), false, 'the stalest lesson falls off');

  assert.deepEqual(mergeLessons(null, null).length, 0);
  const capped = mergeLessons([], [`${'x'.repeat(400)}`], { ts: '2026-09-09T00:00:00.000Z' });
  assert.ok(capped[0].text.length <= LESSON_TEXT_CAP, 'a lesson is one sentence');
});

// --------------------------------------------------------------- rendering

test('lessonsToPrompt renders a bounded ASCII block', () => {
  assert.equal(lessonsToPrompt(null), '');
  assert.equal(lessonsToPrompt([]), '');
  assert.equal(lessonsToPrompt({ lessons: [] }), '');

  const rendered = lessonsToPrompt([
    { text: 'Stage l1 skips in this repo: no lint command detected.' },
    { text: 'The run tool refuses "npm install" in this repo; pick an allowed command instead.' },
  ]);
  const lines = rendered.split('\n');
  assert.equal(lines[0], 'LESSONS FROM EARLIER RUNS IN THIS REPO');
  assert.equal(lines[1], '- Stage l1 skips in this repo: no lint command detected.');
  assert.equal(lines.length, 3);
  assert.equal(/[^\x09\x0A\x20-\x7E]/.test(rendered), false, 'the block is ASCII');

  const many = [];
  for (let i = 0; i < 40; i += 1) many.push({ text: `Lesson ${i}: ${'a fact about this repository. '.repeat(3)}` });
  const big = lessonsToPrompt(many);
  assert.ok(Buffer.byteLength(big, 'utf8') <= LESSONS_BYTE_CAP, `${Buffer.byteLength(big, 'utf8')} bytes`);
  assert.ok(big.startsWith('LESSONS FROM EARLIER RUNS IN THIS REPO\n- Lesson 0'));
});

test('lessonsToPrompt drops a header it cannot fill', () => {
  const oversized = [{ text: 'x'.repeat(LESSON_TEXT_CAP) }];
  const rendered = lessonsToPrompt(oversized);
  // One lesson still fits under 1.5 KB, so this renders; the guard is that a
  // header never ships alone.
  assert.notEqual(rendered, 'LESSONS FROM EARLIER RUNS IN THIS REPO');
  assert.ok(rendered.includes('\n- '));
});

// -------------------------------------------------------------- extraction

const RUN_EVENTS = [
  { type: 'plan', payload: { test_command: 'node --test tests/unit' } },
  {
    type: 'repair',
    payload: { candidateId: 'c1', reasons: ['tests/api.test.js fails at baseline and after the patch'] },
  },
  {
    type: 'repair',
    payload: { candidateId: 'c2', reasons: ['tests/api.test.js fails at baseline and after the patch'] },
  },
  {
    type: 'repair',
    payload: { candidateId: 'c1', reasons: ['a one-off reason only this candidate hit'] },
  },
  {
    type: 'coder.turn',
    payload: {
      detail: [
        { name: 'run', input: 'npm install', result: 'refused: run only executes the detected commands' },
        { name: 'run', input: 'npm test --silent', result: 'exit 0' },
        { name: 'edit_file', input: {}, result: 'refused: old_string is not present in src/legacy/report.js (0 occurrences). Read the file again and copy the exact text, including whitespace.' },
      ],
    },
  },
  {
    type: 'coder.turn',
    payload: {
      detail: [
        { name: 'run', input: 'npx playwright test', result: 'timed out after 120000 ms and was killed (exit code: null); do not rerun it' },
      ],
    },
  },
  { type: 'stage.result', payload: { stage: 'l1', decision: 'skip', reasons: ['no lint command detected for this repo'] } },
  { type: 'stage.result', payload: { stage: 'l2', decision: 'pass', reasons: [] } },
];

test('extractLessonCandidates proposes lessons from hard evidence only', () => {
  const lessons = extractLessonCandidates(RUN_EVENTS);
  assert.ok(lessons.length > 0);
  assert.ok(lessons.length <= LESSON_CANDIDATE_CAP, 'at most five per run');
  const joined = lessons.join('\n');
  assert.match(joined, /Repairs on 2 candidates/);
  assert.match(joined, /refuses "npm install"/);
  assert.match(joined, /npx playwright test.*timed out/);
  assert.match(joined, /src\/legacy\/report\.js/);
  assert.equal(joined.includes('a one-off reason'), false, 'a reason only one candidate hit is not a repo fact');
  for (const lesson of lessons) {
    assert.ok(lesson.length <= LESSON_TEXT_CAP, `"${lesson}" is ${lesson.length} chars`);
    assert.equal(/[^\x09\x20-\x7E]/.test(lesson), false, 'lessons are one ASCII line');
  }
});

test('extractLessonCandidates records the plan test command and a baseline skip', () => {
  const planOnly = extractLessonCandidates([
    { type: 'plan', payload: { test_command: 'node --test tests/unit' } },
    { type: 'stage.result', payload: { stage: 'l1', decision: 'skip', reasons: ['no lint command detected for this repo'] } },
    { type: 'stage.result', payload: { stage: 'l1', decision: 'skip', reasons: ['no lint command detected for this repo'] } },
  ]);
  assert.equal(planOnly.length, 2, 'one lesson per stage, however often it repeats');
  assert.match(planOnly[0], /test command to "node --test tests\/unit"/);
  assert.match(planOnly[1], /Stage l1 skips in this repo/);
});

test('extractLessonCandidates reads raw events as well as stored rows', () => {
  const raw = extractLessonCandidates([
    { type: 'coder.turn', data: { detail: [{ name: 'run', input: 'make bench', result: 'refused: not an allowed command' }] } },
  ]);
  assert.equal(raw.length, 1);
  assert.match(raw[0], /refuses "make bench"/);
});

test('extractLessonCandidates returns nothing for a clean run', () => {
  assert.deepEqual(extractLessonCandidates([]), []);
  assert.deepEqual(extractLessonCandidates(null), []);
  assert.deepEqual(extractLessonCandidates([
    { type: 'stage.result', payload: { stage: 'l2', decision: 'pass', reasons: [] } },
    { type: 'coder.turn', payload: { detail: [{ name: 'read_file', input: 'src/a.js', result: 'ok' }] } },
  ]), []);
});

test('a run feeds back into the file the next run reads', () => {
  const dir = tempDir('cycle');
  const config = fakeConfig(dir);
  const first = mergeLessons(loadLessons(config, REPO), extractLessonCandidates(RUN_EVENTS), { runId: 'run-1', ts: '2026-09-08T00:00:00.000Z' });
  saveLessons(config, REPO, first);

  const second = mergeLessons(loadLessons(config, REPO), extractLessonCandidates(RUN_EVENTS), { runId: 'run-2', ts: '2026-09-09T00:00:00.000Z' });
  saveLessons(config, REPO, second);

  const doc = loadLessons(config, REPO);
  assert.equal(doc.lessons.length, first.length, 'a repeat run adds no duplicates');
  assert.ok(doc.lessons.every((l) => l.hits === 2), 'every lesson was seen twice');
  assert.ok(lessonsToPrompt(doc).includes('LESSONS FROM EARLIER RUNS IN THIS REPO'));
  fs.rmSync(dir, { recursive: true, force: true });
});
