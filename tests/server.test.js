import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { rmSyncBestEffort } from './_spawn-helpers.js';

test('server trusts the two-hop Vercel + AWS ALB edge chain in production', async (t) => {
  const savedEnv = {
    NODE_ENV: process.env.NODE_ENV,
    AWS_EXECUTION_ENV: process.env.AWS_EXECUTION_ENV,
    VERCEL: process.env.VERCEL,
    AWS_LAMBDA_FUNCTION_NAME: process.env.AWS_LAMBDA_FUNCTION_NAME,
    KOLM_DATA_DIR: process.env.KOLM_DATA_DIR,
  };
  const testDataDir = path.join(os.tmpdir(), `kolm-server-${process.pid}-${Date.now()}`);

  process.env.KOLM_DATA_DIR = testDataDir;
  delete process.env.NODE_ENV;
  process.env.AWS_EXECUTION_ENV = 'AWS_ECS_FARGATE';
  delete process.env.VERCEL;
  delete process.env.AWS_LAMBDA_FUNCTION_NAME;

  t.after(() => {
    for (const [key, value] of Object.entries(savedEnv)) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
    rmSyncBestEffort(testDataDir);
  });

  const { app } = await import(`../server.js?trust-proxy=${Date.now()}`);
  // Two-hop chain: Client -> Vercel CDN -> AWS ALB -> Express. Express
  // must trust both hops so rate-limit keyGenerator and req.ip resolve to the
  // real client IP, not the ALB's IP.
  assert.equal(app.get('trust proxy'), 2);
});

test('artifact receipts label the HMAC signature algorithm truthfully', async (t) => {
  const savedSecret = process.env.RECIPE_RECEIPT_SECRET;
  const savedEd25519 = process.env.KOLM_ED25519_DISABLE;
  const savedSigstore = process.env.KOLM_SIGSTORE_DISABLE;
  process.env.RECIPE_RECEIPT_SECRET = 'test_receipt_secret';
  // Wave 149/150 — this contract test pins signature_alg to 'hmac-sha256',
  // which is the legacy mode. Disable Ed25519 and sigstore so the test
  // exercises HMAC-only signing without removing those defaults globally.
  process.env.KOLM_ED25519_DISABLE = '1';
  process.env.KOLM_SIGSTORE_DISABLE = '1';
  t.after(() => {
    if (savedSecret === undefined) delete process.env.RECIPE_RECEIPT_SECRET;
    else process.env.RECIPE_RECEIPT_SECRET = savedSecret;
    if (savedEd25519 === undefined) delete process.env.KOLM_ED25519_DISABLE;
    else process.env.KOLM_ED25519_DISABLE = savedEd25519;
    if (savedSigstore === undefined) delete process.env.KOLM_SIGSTORE_DISABLE;
    else process.env.KOLM_SIGSTORE_DISABLE = savedSigstore;
  });

  const { buildPayload } = await import(`../src/artifact.js?receipt-alg=${Date.now()}`);
  const payload = buildPayload({
    job_id: 'job_test',
    task: 'classify support tickets',
    recipes: [{ id: 'recipe_test', name: 'test', source: 'function generate(){return true}', source_hash: 'sha256:test' }],
    training_stats: { distilled_pairs: 0, accuracy: null },
    evals: { spec: 'rs-1-evals', n: 1, cases: [{ input: 'hello', expected: true }] },
  });

  assert.equal(payload.receipt.signature_alg, 'hmac-sha256');
  assert.match(payload.receipt.signature, /^[0-9a-f]{64}$/);
  assert.deepEqual(payload.receipt.anchors, []);
});

test('artifact benchmark emits reproducible local report', async (t) => {
  const savedSecret = process.env.RECIPE_RECEIPT_SECRET;
  const savedEd25519 = process.env.KOLM_ED25519_DISABLE;
  const savedSigstore = process.env.KOLM_SIGSTORE_DISABLE;
  process.env.RECIPE_RECEIPT_SECRET = 'ks_receipt_' + 'c'.repeat(48);
  // Wave 149/150 — this benchmark test pins receipt_signature_alg to
  // 'hmac-sha256' (legacy). Disable Ed25519 and sigstore so the test
  // exercises HMAC-only signing without removing those defaults globally.
  process.env.KOLM_ED25519_DISABLE = '1';
  process.env.KOLM_SIGSTORE_DISABLE = '1';
  const outDir = path.join(os.tmpdir(), `kolm-benchmark-${process.pid}-${Date.now()}`);
  fs.mkdirSync(outDir, { recursive: true });
  t.after(() => {
    if (savedSecret === undefined) delete process.env.RECIPE_RECEIPT_SECRET;
    else process.env.RECIPE_RECEIPT_SECRET = savedSecret;
    if (savedEd25519 === undefined) delete process.env.KOLM_ED25519_DISABLE;
    else process.env.KOLM_ED25519_DISABLE = savedEd25519;
    if (savedSigstore === undefined) delete process.env.KOLM_SIGSTORE_DISABLE;
    else process.env.KOLM_SIGSTORE_DISABLE = savedSigstore;
    rmSyncBestEffort(outDir);
  });

  const { buildAndZip } = await import(`../src/artifact.js?benchmark-build=${Date.now()}`);
  const built = await buildAndZip({
    job_id: 'job_bench',
    task: 'classify message urgency',
    recipes: [{
      id: 'recipe_urgency',
      name: 'urgency',
      source: 'function generate(input){ return String(input.text || "").includes("urgent") ? "urgent" : "normal"; }',
      source_hash: 'sha256:bench',
    }],
    training_stats: { verifier_accepted: true, pass_rate_positive: 1, latency_p50_us: 50 },
    evals: {
      spec: 'rs-1-evals',
      n: 2,
      cases: [
        { id: 'urgent', input: { text: 'urgent chest pain' }, expected: 'urgent' },
        { id: 'normal', input: { text: 'billing question' }, expected: 'normal' },
      ],
      coverage: 1,
    },
    outDir,
  });

  const { benchmarkArtifact } = await import(`../src/benchmark.js?benchmark=${Date.now()}`);
  const reportPath = path.join(outDir, 'benchmark.json');
  const report = await benchmarkArtifact(built.outPath, {
    runs: 2,
    target: 'test-target',
    device: 'test-device',
    outPath: reportPath,
  });

  assert.equal(report.spec, 'kolm-benchmark-1');
  assert.equal(report.artifact_sha256.length, 'sha256:'.length + 64);
  assert.equal(report.target, 'test-target');
  assert.equal(report.device, 'test-device');
  assert.equal(report.evals.n, 2);
  assert.equal(report.evals.graded, 4);
  assert.equal(report.evals.passed, 4);
  assert.equal(report.evals.accuracy, 1);
  assert.equal(report.latency_us.n, 4);
  assert.equal(report.privacy.runtime_egress_attempts, 0);
  assert.equal(report.integrity.signature_valid, true);
  assert.equal(report.integrity.receipt_present, true);
  assert.equal(report.integrity.receipt_signature_alg, 'hmac-sha256');
  assert.ok(fs.existsSync(reportPath), 'benchmark should write requested report path');

  const cli = spawnSync(process.execPath, ['cli/kolm.js', 'benchmark', built.outPath, '--runs', '1', '--target', 'cli-target'], {
    cwd: path.resolve('.'),
    env: { ...process.env, RECIPE_RECEIPT_SECRET: process.env.RECIPE_RECEIPT_SECRET },
    encoding: 'utf8',
  });
  assert.equal(cli.status, 0, cli.stderr);
  const cliReport = JSON.parse(cli.stdout);
  assert.equal(cliReport.spec, 'kolm-benchmark-1');
  assert.equal(cliReport.target, 'cli-target');
  assert.equal(cliReport.evals.passed, 2);
});
