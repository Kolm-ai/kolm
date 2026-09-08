// Owner D — the signed run receipt (spec 6.5): build, sign, verify, tamper,
// canonical stability, and the hard rule that no credential-shaped text ever
// reaches the file. No network, no store, no provider: buildReceipt is pure.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import {
  buildReceipt,
  signReceipt,
  verifyReceipt,
  writeReceipt,
  readReceipt,
  receiptPathFor,
  scrubSecrets,
  RECEIPT_SCHEMA,
} from '../src/ladder/receipt.js';
import { generateKeyPair, keyFingerprint } from '../src/ed25519.js';
import { canonicalJson } from '../src/cid.js';
import { loadBands } from '../src/ladder/bands.js';
import { loadRubric } from '../src/ladder/rubric.js';

const BANDS = loadBands();
const RUBRIC = loadRubric();

/** A signer that lives only in this process: no ~/.kolm side effects. */
function testSigner() {
  const { publicKey, privateKey } = generateKeyPair();
  return { publicKey, privateKey, key_fingerprint: keyFingerprint(publicKey) };
}

const tempDirs = [];
function tempDir(label) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), `kolm-receipt-${label}-`));
  tempDirs.push(dir);
  return dir;
}

test.after(() => {
  for (const dir of tempDirs) {
    try {
      fs.rmSync(dir, { recursive: true, force: true });
    } catch {
      // a lingering handle on Windows is not a test failure
    }
  }
});

// --------------------------------------------------------------- fixtures

const RUN_ROW = {
  id: 'run_20260101_abcdef',
  task: { title: 'add add()', text: 'add a function add(a,b) to src/math.js' },
  repo_path: 'C:\\Users\\someone\\repos\\demo',
  base_sha: 'a'.repeat(40),
  mode: 'cascade',
};

const CANDIDATE_ROWS = [
  { candidate_id: 'c0a0', generator: 'anthropic:claude-haiku', verdict: 'PASS', diff_sha256: 'd'.repeat(64) },
  { candidate_id: 'c1a0', generator: 'anthropic:claude-haiku', verdict: 'FAIL', diff_sha256: 'e'.repeat(64) },
];

const RESULT_ROWS = [
  {
    candidate_id: 'c1a0',
    stage: 'l2-dynamic',
    version: 1,
    verdict: { decision: 'bad', reasons: ['1 newly failing test(s): adds numbers'], metrics: { newly_failing_tests: 1 } },
  },
  {
    candidate_id: 'c0a0',
    stage: 'l0-meta',
    version: 1,
    verdict: { decision: 'good', reasons: [], metrics: { patch_lines: 12, files_touched: 2 } },
  },
  {
    candidate_id: 'c0a0',
    stage: 'l2-dynamic',
    version: 1,
    verdict: { decision: 'good', reasons: ['no newly failing tests (3 passed, 0 failed)'], metrics: { passed: 3 } },
  },
];

const JUDGMENT_ROWS = [
  {
    run_id: RUN_ROW.id,
    candidate_id: 'c0a0',
    kind: 'unsure',
    verdict: { decision: 'good', score: 0.91, blocking: [], rationale: 'the patch adds the function and a test for it' },
  },
];

const COST = {
  total_usd: 0.0421,
  by_model: { 'anthropic:claude-haiku': { cost_usd: 0.0121 }, 'anthropic:claude-opus': { cost_usd: 0.03 } },
};

const MODELS = {
  orchestrator: 'anthropic:claude-sonnet',
  judge: 'anthropic:claude-opus',
  coders: ['anthropic:claude-haiku', 'anthropic:claude-sonnet'],
  reviewer: 'anthropic:claude-haiku',
};

const CONFIG = {
  candidates: 3,
  coderTurns: 24,
  repairTurns: 2,
  maxCostUsd: 5,
  calibrationRate: 0.1,
  codeDir: path.join('data', 'code'),
};

function passingReceipt(overrides = {}) {
  return buildReceipt({
    run: RUN_ROW,
    candidates: CANDIDATE_ROWS,
    results: RESULT_ROWS,
    judgments: JUDGMENT_ROWS,
    cost: COST,
    winner: 'c0a0',
    branch: `kolm/code/${RUN_ROW.id}/c0a0`,
    sha: 'b'.repeat(40),
    mode: 'cascade',
    bands: BANDS,
    rubric: RUBRIC,
    config: CONFIG,
    models: MODELS,
    calibration: { sampled: 2, agreementRate: 1 },
    stopReason: 'winner',
    verdict: 'PASS',
    createdAt: '2026-01-01T00:00:00.000Z',
    ...overrides,
  });
}

// ------------------------------------------------------------------ shape

test('receipt carries the spec 6.5 shape and never the raw diff', () => {
  const receipt = passingReceipt();
  assert.equal(receipt.schema, RECEIPT_SCHEMA);
  assert.equal(receipt.version, 1);
  assert.equal(receipt.run_id, RUN_ROW.id);
  assert.equal(receipt.mode, 'cascade');
  assert.equal(receipt.verdict, 'PASS');
  assert.equal(receipt.stop_reason, 'winner');
  assert.equal(receipt.task.title, 'add add()');
  assert.equal(receipt.task.text_sha256.length, 64);
  assert.equal(receipt.repo.base_sha, RUN_ROW.base_sha);
  assert.equal(receipt.repo.path_sha256.length, 64);
  assert.equal(receipt.winner.candidate_id, 'c0a0');
  assert.equal(receipt.winner.diff_sha256, 'd'.repeat(64));
  assert.equal(receipt.candidates.length, 2);
  assert.equal(receipt.stages.length, 3);
  assert.equal(receipt.judge.candidate_id, 'c0a0');
  assert.equal(receipt.judge.decision, 'good');
  assert.equal(receipt.judge.rationale_sha256.length, 64);
  assert.equal(receipt.judge.rationale, undefined);
  assert.equal(receipt.calibration.sampled, 2);
  assert.equal(receipt.cost.total_usd, 0.0421);
  assert.deepEqual(receipt.models.coders, MODELS.coders);
  assert.equal(receipt.ladder_config.candidates, 3);
  assert.equal(receipt.ladder_config.bands_sha256.length, 64);
  assert.equal(receipt.ladder_config.rubric_sha256.length, 64);
  // Stage rows are sorted by (candidate, stage) so the canonical form is stable.
  assert.deepEqual(
    receipt.stages.map((s) => `${s.candidate_id}/${s.stage}`),
    ['c0a0/l0-meta', 'c0a0/l2-dynamic', 'c1a0/l2-dynamic'],
  );
});

// ------------------------------------------------------- sign / verify

test('signing then verifying a receipt round trips', () => {
  const signer = testSigner();
  const signed = signReceipt(passingReceipt(), signer);
  assert.equal(signed.signer_kid, signer.key_fingerprint);
  assert.ok(signed.signature_ed25519, 'a signature block is attached');
  const result = verifyReceipt(signed);
  assert.equal(result.ok, true, result.reason ?? 'no reason given');
  assert.equal(result.kid, signer.key_fingerprint);
  assert.equal(result.reason, null);
});

test('pinning a different public key rejects the receipt', () => {
  const signed = signReceipt(passingReceipt(), testSigner());
  const other = testSigner();
  const result = verifyReceipt(signed, { publicKey: other.publicKey });
  assert.equal(result.ok, false);
  assert.match(result.reason, /pinned key/);
});

test('tampering with any signed field fails verification', () => {
  const signer = testSigner();
  const signed = signReceipt(passingReceipt(), signer);

  const mutations = {
    verdict: (r) => { r.verdict = 'FAIL'; },
    'cost.total_usd': (r) => { r.cost.total_usd = 0.0001; },
    'winner.sha': (r) => { r.winner.sha = 'c'.repeat(40); },
    'stages[0].decision': (r) => { r.stages[0].decision = 'bad'; },
    'judge.score': (r) => { r.judge.score = 0.1; },
    run_id: (r) => { r.run_id = 'run_forged'; },
    signer_kid: (r) => { r.signer_kid = 'deadbeef'; },
    'models.coders': (r) => { r.models.coders.push('anthropic:claude-opus'); },
  };

  for (const [label, mutate] of Object.entries(mutations)) {
    const copy = JSON.parse(JSON.stringify(signed));
    mutate(copy);
    const result = verifyReceipt(copy);
    assert.equal(result.ok, false, `mutation of ${label} must not verify`);
    assert.ok(typeof result.reason === 'string' && result.reason.length > 0);
  }
});

test('unsigned receipts report themselves as unsigned instead of passing', () => {
  const unsigned = signReceipt(passingReceipt(), null);
  assert.equal(unsigned.signature_ed25519, undefined);
  const result = verifyReceipt(unsigned);
  assert.equal(result.ok, false);
  assert.match(result.reason, /not signed/);
});

// ------------------------------------------------------ canonical order

test('canonical form does not depend on input ordering', () => {
  const forward = passingReceipt();
  const shuffled = passingReceipt({
    results: [...RESULT_ROWS].reverse(),
    candidates: [...CANDIDATE_ROWS],
  });
  assert.equal(canonicalJson(forward), canonicalJson(shuffled));

  // And signing the two produces byte-identical payloads.
  const signer = testSigner();
  const a = signReceipt(forward, signer);
  const b = signReceipt(shuffled, signer);
  delete a.signature_ed25519;
  delete b.signature_ed25519;
  assert.equal(canonicalJson(a), canonicalJson(b));
});

test('every absent value is an explicit null, never undefined', () => {
  const receipt = buildReceipt({ run: { id: 'run_bare' } });
  const text = JSON.stringify(receipt);
  assert.ok(!text.includes('undefined'));
  assert.equal(receipt.winner, null);
  assert.equal(receipt.judge, null);
  assert.equal(receipt.repo.path, null);
  assert.equal(receipt.repo.path_sha256, null);
  assert.equal(receipt.task.text_sha256, null);
  assert.equal(receipt.calibration.agreement_rate, null);
  // canonicalJson refuses to represent undefined, so it must survive a round trip.
  assert.equal(JSON.parse(canonicalJson(receipt)).run_id, 'run_bare');
});

// ------------------------------------------------------------- secrets

test('credential-shaped text never survives into a receipt', () => {
  const leaky = [
    { candidate_id: 'c0a0', stage: 'l0-meta', version: 1, verdict: {
      decision: 'bad',
      reasons: [
        'patch adds a key sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAA',
        'a github token ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ012345 appeared',
        'aws id AKIAIOSFODNN7EXAMPLE in config',
        'header was Bearer eyJhbGciOiJIUzI1NiJ9.payload',
        'slack token xoxb-1234567890-abcdefghij',
      ],
      metrics: { note: 'context: sk-ant-api03-BBBBBBBBBBBBBBBBBBBBBBBB', secrets: 5 },
    } },
  ];
  const receipt = buildReceipt({
    run: { id: 'run_leaky', task: { title: 'token sk-ant-api03-CCCCCCCCCCCCCCCCCCCCCCCC', text: 'secret body' } },
    results: leaky,
    judgments: [{ candidate_id: 'c0a0', kind: 'unsure', verdict: { decision: 'bad', rationale: 'ghp_ZZZZZZZZZZZZZZZZZZZZZZZZ' } }],
    cost: COST,
    bands: BANDS,
    rubric: RUBRIC,
    config: CONFIG,
    verdict: 'FAIL',
  });
  const serialized = JSON.stringify(signReceipt(receipt, testSigner()));
  for (const pattern of [/sk-ant-/, /ghp_/, /AKIA[0-9A-Z]{12}/, /Bearer\s+\S{8}/, /xoxb-\d/]) {
    assert.ok(!pattern.test(serialized), `pattern ${pattern} must not appear in the receipt`);
  }
  assert.ok(serialized.includes('[redacted]'));
  // The judge rationale is stored only as a hash, so it cannot leak at all.
  assert.equal(receipt.judge.rationale_sha256.length, 64);
  assert.ok(!serialized.includes('secret body'));
});

test('scrubSecrets redacts each known credential shape', () => {
  const cleaned = scrubSecrets('a sk-ant-api03-QQQQQQQQQQQQQQQQ and ghp_AAAAAAAAAAAAAAAAAAAA and AKIAIOSFODNN7EXAMPLE');
  assert.ok(!cleaned.includes('sk-ant-'));
  assert.ok(!cleaned.includes('ghp_'));
  assert.ok(!cleaned.includes('AKIAIOSFODNN7EXAMPLE'));
  assert.equal(scrubSecrets(null), '');
});

// -------------------------------------------------------- failing runs

test('a FAIL receipt with a null winner still verifies', () => {
  const signer = testSigner();
  const receipt = buildReceipt({
    run: RUN_ROW,
    candidates: [{ candidate_id: 'c0a0', generator: 'ollama:qwen', verdict: 'FAIL' }],
    results: RESULT_ROWS,
    judgments: [],
    cost: { total_usd: 0.01, by_model: {} },
    winner: null,
    branch: null,
    sha: null,
    mode: 'cascade',
    bands: BANDS,
    rubric: RUBRIC,
    config: CONFIG,
    models: MODELS,
    calibration: { sampled: 0, agreementRate: null },
    stopReason: 'ladder-exhausted',
    verdict: 'FAIL',
  });
  assert.equal(receipt.winner, null);
  assert.equal(receipt.verdict, 'FAIL');
  assert.equal(receipt.stop_reason, 'ladder-exhausted');
  const result = verifyReceipt(signReceipt(receipt, signer));
  assert.equal(result.ok, true, result.reason ?? 'no reason given');
});

// ------------------------------------------------------------ file I/O

test('writeReceipt and readReceipt round trip through a file', () => {
  const dir = tempDir('write');
  const signed = signReceipt(passingReceipt(), testSigner());
  const written = writeReceipt(signed, path.join(dir, 'receipts'));
  assert.equal(path.basename(written), `${RUN_ROW.id}.json`);
  assert.ok(fs.existsSync(written));
  const loaded = readReceipt(written);
  assert.deepEqual(loaded, JSON.parse(JSON.stringify(signed)));
  const fromPath = verifyReceipt(written);
  assert.equal(fromPath.ok, true, fromPath.reason ?? 'no reason given');
});

test('verifyReceipt reports a missing file instead of throwing', () => {
  const result = verifyReceipt(path.join(tempDir('missing'), 'nope.json'));
  assert.equal(result.ok, false);
  assert.match(result.reason, /cannot read receipt/);
});

test('receiptPathFor places receipts under the code directory', () => {
  const target = receiptPathFor({ codeDir: path.join('data', 'code') }, 'run_xyz');
  assert.equal(target, path.join('data', 'code', 'receipts', 'run_xyz.json'));
});
