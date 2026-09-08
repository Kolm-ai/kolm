import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';

const ROOT = path.resolve('.');
const MATRIX_PATH = path.join(ROOT, 'docs', 'internal', 'cli-command-matrix.json');

function readJson(relOrAbs) {
  return JSON.parse(fs.readFileSync(path.isAbsolute(relOrAbs) ? relOrAbs : path.join(ROOT, relOrAbs), 'utf8'));
}

function matrix() {
  assert.ok(fs.existsSync(MATRIX_PATH), 'cli-command-matrix.json must exist');
  return readJson(MATRIX_PATH);
}

test('W939 package wiring makes the CLI command matrix a depth and control-file gate', () => {
  const pkg = readJson('package.json');
  assert.equal(pkg.scripts['build:cli-command-matrix'], 'node scripts/build-cli-command-matrix.cjs');
  assert.equal(
    pkg.scripts['verify:cli-command-matrix'],
    'node scripts/build-cli-command-matrix.cjs --check --summary && node --test --test-concurrency=1 tests/wave939-cli-command-matrix.test.js',
  );
  assert.match(pkg.scripts['build:control-files'], /build:auth-boundary-matrix && npm run build:cli-command-matrix && npm run build:daemon-connector-matrix && npm run build:quantize-worker-matrix && npm run build:binder-contract-matrix && npm run build:intent-contract-matrix && npm run build:wrapper-cli-matrix && npm run build:distill-pipeline-matrix && npm run build:spec-compile-matrix && npm run build:data-curate-matrix && npm run build:artifact-matrix && npm run build:tui-workbench-matrix && npm run build:bench-harness-matrix && npm run build:otel-matrix && npm run build:readiness-proof-matrix && npm run build:frontier-delta-freshness && npm run build:file-ledger/);
  assert.match(pkg.scripts['verify:control-files'], /verify:auth-boundary-matrix && npm run verify:cli-command-matrix && npm run verify:daemon-connector-matrix && npm run verify:quantize-worker-matrix && npm run verify:binder-contract-matrix && npm run verify:intent-contract-matrix && npm run verify:wrapper-cli-matrix && npm run verify:distill-pipeline-matrix && npm run verify:spec-compile-matrix && npm run verify:data-curate-matrix && npm run verify:artifact-matrix && npm run verify:tui-workbench-matrix && npm run verify:bench-harness-matrix && npm run verify:otel-matrix && npm run verify:readiness-proof-matrix && npm run verify:frontier-delta-freshness && npm run verify:file-ledger/);
  assert.match(pkg.scripts['verify:depth'], /verify:auth-boundary-matrix && npm run verify:cli-command-matrix && npm run verify:daemon-connector-matrix && npm run verify:binder-contract-matrix && npm run verify:intent-contract-matrix && npm run verify:wrapper-cli-matrix && npm run verify:distill-pipeline-matrix && npm run verify:spec-compile-matrix && npm run verify:data-curate-matrix && npm run verify:artifact-matrix && npm run verify:tui-workbench-matrix && npm run verify:bench-harness-matrix && npm run verify:otel-matrix && npm run verify:readiness-proof-matrix && npm run verify:meta-routes/);

  const ledgerBuilder = fs.readFileSync(path.join(ROOT, 'scripts', 'build-codebase-file-ledger.cjs'), 'utf8');
  assert.match(ledgerBuilder, /build-cli-command-matrix\.cjs/);
  assert.match(ledgerBuilder, /docs\/internal\/cli-command-matrix\.json/);

  const releaseVerify = fs.readFileSync(path.join(ROOT, 'scripts', 'release-verify.cjs'), 'utf8');
  assert.match(releaseVerify, /docs\/internal\/cli-command-matrix\.json/);
  assert.match(releaseVerify, /kolm\.cli_command_matrix\.v1/);

  const backendAtomic = fs.readFileSync(path.join(ROOT, 'scripts', 'build-backend-atomic-deep-dive.mjs'), 'utf8');
  assert.match(backendAtomic, /CLI_COMMAND_MATRIX/);
  assert.match(backendAtomic, /maintain_generated_cli_command_matrix_and_split_plan/);
  assert.match(backendAtomic, /npm run verify:cli-command-matrix/);
});

test('W939 generated matrix is current and all hard CLI command gates are green', () => {
  execFileSync(process.execPath, ['scripts/build-cli-command-matrix.cjs', '--check', '--summary'], {
    cwd: ROOT,
    stdio: 'pipe',
    timeout: 60_000,
  });
  const m = matrix();

  assert.equal(m.schema, 'kolm.cli_command_matrix.v1');
  assert.equal(m.updated_at, '2026-06-18');
  assert.equal(m.gates.ok, true, JSON.stringify(m.gates.failures, null, 2));
  assert.deepEqual(m.gates.failures, []);
  assert.equal(m.summary.dispatcher_case_count, 259);
  assert.equal(m.summary.command_function_count, 334);
  assert.equal(m.summary.completion_verb_count, 255);
  assert.equal(m.summary.product_graph_cli_commands, 64);
  assert.equal(m.summary.product_graph_cli_verbs, 40);
  assert.equal(m.summary.product_graph_proof_cli_commands, 54);
  assert.equal(m.summary.missing_product_graph_verbs, 0);
  assert.equal(m.summary.missing_product_graph_proof_verbs, 0);
  assert.equal(m.summary.completion_without_dispatch, 0);
  assert.equal(m.summary.dispatch_without_completion, 0);
  assert.equal(m.summary.failed_safety_guards, 0);
  assert.equal(m.summary.missing_test_evidence, 0);
  assert.deepEqual(m.gates.warnings.map((w) => w.gate), ['public_cli_docs_directory_absent']);
});

test('W939 matrix captures product graph CLI commands and dispatcher families', () => {
  const m = matrix();
  const cases = new Map(m.dispatcher_cases.map((row) => [row.verb, row]));
  const commandFunctions = new Set(m.command_functions.map((row) => row.name));
  assert.ok(commandFunctions.has('cmdRouteRetrain'), 'route retrain handler must be in the CLI command matrix');
  assert.ok(commandFunctions.has('cmdRouteRollback'), 'route rollback handler must be in the CLI command matrix');
  for (const verb of ['compile', 'run', 'serve', 'cloud', 'compute', 'devices', 'pipeline', 'receipts', 'packages', 'audio', 'video', 'vlm']) {
    assert.ok(cases.has(verb), `dispatcher missing ${verb}`);
    assert.equal(cases.get(verb).in_completion_verbs, true, `${verb} must be in completion verbs`);
  }

  const productCommands = new Set(m.product_graph_cli_commands.map((row) => row.command));
  for (const command of [
    'kolm compile --spec spec.json --out task.kolm',
    'kolm run task.kolm "input"',
    'kolm cloud broker --json',
    'kolm devices detect --json',
    'kolm keys list',
  ]) {
    assert.ok(productCommands.has(command), `product graph CLI command missing: ${command}`);
  }
  assert.equal(m.product_graph_cli_commands.every((row) => row.dispatch_present), true);
  assert.equal(m.product_graph_proof_commands.every((row) => row.dispatch_present), true);

  for (const family of ['developer_distribution', 'compile_artifact', 'runtime_serving', 'capture_data', 'training_eval', 'infra_device', 'governance_security']) {
    assert.ok(m.dispatcher_family_counts[family] > 0, `family ${family} must be represented`);
  }
});

test('W939 matrix captures CLI safety guards and direct test evidence', () => {
  const m = matrix();
  assert.equal(Object.values(m.safety_guards).every(Boolean), true);
  assert.equal(m.package_bin.kolm, 'cli/kolm.js');

  const evidence = new Map(m.test_evidence.map((row) => [row.path, row.present]));
  for (const rel of [
    'tests/finalized-c11-cli-tui-dx-contract.test.js',
    'tests/wave921-cli-dx.test.js',
    'tests/sota-cli.test.js',
    'tests/wrapper-integration.test.js',
    'tests/wrapper-smoke.test.js',
  ]) {
    assert.equal(evidence.get(rel), true, `${rel} must be direct CLI evidence`);
  }
});
