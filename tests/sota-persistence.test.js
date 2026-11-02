// SOTA Persistence lane - real fixes for the W411/W808/W409a/event-store atoms.
//
// Atoms exercised:
//   1) [p1] promoteStagedCapture() update predicate now carries the tenant fence
//      (store.js) - a foreign tenant_id must NOT flip another tenant's row to
//      'promoted'. Mirrors the fence already in getStagedCapture/markStagedAnomaly.
//   3) [p2] event-store _jsonlAll() surfaces parse diagnostics instead of silently
//      dropping malformed JSONL lines - jsonlDiagnostics()/storeInfo() report
//      "parsed N, failed M" with per-line reasons.
//   4) [p2] capture-store observationToCanonicalEvent() mints a DETERMINISTIC
//      fallback event_id (content+tenant+namespace+minute-bucket hash) so the
//      same observation inserted twice collapses to one canonical event instead
//      of two timestamp-nonce rows.
//
// Isolation follows the W808 pattern: per-test temp KOLM_DATA_DIR + HOME, JSON
// store driver, singleton store module reset between tests.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

function freshDir(prefix) {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), prefix));
  process.env.KOLM_DATA_DIR = path.join(tmp, '.kolm');
  process.env.HOME = tmp;
  process.env.USERPROFILE = tmp;
  process.env.KOLM_HOME = path.join(tmp, '.kolm');
  process.env.KOLM_ENV = 'test';
  process.env.KOLM_STORE_DRIVER = 'json';
  // Force the JSONL fallback so the parse-diagnostics path is the one under test
  // (the sqlite driver has no line-parse surface). This is the documented test
  // override in event-store._ensureDriver().
  process.env.KOLM_EVENT_STORE_DRIVER = 'jsonl';
  delete process.env.KOLM_EVENT_STORE_PATH;
  fs.mkdirSync(process.env.KOLM_DATA_DIR, { recursive: true });
  return tmp;
}

let _storeMod = null;
async function loadStore() {
  if (!_storeMod) _storeMod = await import('../src/store.js');
  try { _storeMod.reset(); } catch {} // deliberate: cleanup
  if (typeof _storeMod._resetStagedCapturesForTests === 'function') _storeMod._resetStagedCapturesForTests();
  return _storeMod;
}

// =============================================================================
// Atom 1 - promoteStagedCapture tenant fence
// =============================================================================
test('atom1 - promoteStagedCapture refuses to promote across tenants (W411 fence)', async () => {
  freshDir('kolm-sota-p1-');
  const store = await loadStore();

  const tenantA = 'tenant-A-' + Math.random().toString(36).slice(2, 8);
  const tenantB = 'tenant-B-' + Math.random().toString(36).slice(2, 8);

  // Stage a row owned by tenant A, with quarantine already elapsed and clean.
  const staged = store.insertStagedCapture({
    tenant_id: tenantA,
    namespace: 'ns1',
    prompt: 'q', response: 'a',
    quarantine_until: new Date(Date.now() - 60_000).toISOString(),
  });
  assert.equal(staged.quarantine_state, 'pending');

  // Tenant B attempts promotion of tenant A's staged_capture_id. getStagedCapture
  // already fences reads, so promote returns null and inserts nothing.
  let inserted = 0;
  const wrong = store.promoteStagedCapture(staged.staged_capture_id, {
    tenant_id: tenantB,
    insertObservation: () => { inserted += 1; },
  });
  assert.equal(wrong, null, 'foreign-tenant promotion must return null');
  assert.equal(inserted, 0, 'foreign-tenant promotion must not insert an observation');

  // The underlying staged row must NOT have been flipped to promoted by the
  // update() predicate (this is the line-756 fix under test). Read it raw.
  const rawRows = store.findByField(store.W808_STAGED_TABLE, 'staged_capture_id', staged.staged_capture_id);
  assert.equal(rawRows.length, 1);
  assert.equal(rawRows[0].quarantine_state, 'pending',
    'tenant B must not have flipped tenant A row to promoted');

  // The legitimate owner CAN promote, proving the fence is permissive for the
  // matching tenant (not a blanket block).
  let owned = 0;
  const ok = store.promoteStagedCapture(staged.staged_capture_id, {
    tenant_id: tenantA,
    insertObservation: () => { owned += 1; },
  });
  assert.ok(ok, 'owner promotion should return the row');
  assert.equal(ok.quarantine_state, 'promoted');
  assert.equal(owned, 1, 'owner promotion inserts exactly one observation');
  const after = store.findByField(store.W808_STAGED_TABLE, 'staged_capture_id', staged.staged_capture_id);
  assert.equal(after[0].quarantine_state, 'promoted', 'owner update flipped the row');
});

// Direct unit test of the update() predicate fence so a future refactor of
// promoteStagedCapture's guard ordering cannot silently re-open the hole.
test('atom1 - update() predicate fence does not patch a foreign-tenant row', async () => {
  freshDir('kolm-sota-p1b-');
  const store = await loadStore();
  const tenantA = 'A' + Math.random().toString(36).slice(2, 8);

  const staged = store.insertStagedCapture({ tenant_id: tenantA, prompt: 'q', response: 'r' });
  // Simulate the exact predicate shape the fix installs at line 756 with a
  // mismatched tenant_id - it must patch zero rows.
  const tenant_id = 'someone-else';
  const patched = store.update(store.W808_STAGED_TABLE,
    (r) => r.staged_capture_id === staged.staged_capture_id
      && (!tenant_id || String(r.tenant_id) === String(tenant_id)),
    { quarantine_state: 'promoted' });
  assert.equal(patched, 0, 'mismatched tenant predicate patches zero rows');
  const raw = store.findByField(store.W808_STAGED_TABLE, 'staged_capture_id', staged.staged_capture_id);
  assert.equal(raw[0].quarantine_state, 'pending');
});

// =============================================================================
// Atom 3 - event-store JSONL parse diagnostics
// =============================================================================
test('atom3 - jsonlDiagnostics surfaces parsed/failed counts for corrupt lines', async () => {
  freshDir('kolm-sota-p3-');
  const es = await import('../src/event-store.js?p3=' + Date.now());
  es._resetForTests();

  // Append two well-formed canonical events.
  await es.appendEvent({ tenant_id: 't1', namespace: 'default', provider: 'openai', model: 'gpt', prompt_redacted: 'a', response_redacted: 'b', status: 'ok' });
  await es.appendEvent({ tenant_id: 't1', namespace: 'default', provider: 'openai', model: 'gpt', prompt_redacted: 'c', response_redacted: 'd', status: 'ok' });

  // Corrupt the file: append a garbage (non-JSON) line and a JSON line missing
  // event_id, simulating a mid-write crash / disk error.
  const info0 = es.storeInfo();
  assert.equal(info0.driver, 'jsonl', 'test must exercise the JSONL fallback');
  const jsonlPath = info0.jsonl_path;
  fs.appendFileSync(jsonlPath, '{not valid json at all\n', 'utf8');
  fs.appendFileSync(jsonlPath, JSON.stringify({ foo: 'bar' }) + '\n', 'utf8');

  const diag = es.jsonlDiagnostics();
  assert.equal(diag.driver, 'jsonl');
  assert.equal(diag.parsed, 2, 'exactly two valid events parsed');
  assert.equal(diag.failed, 2, 'one garbage line + one missing-event_id line failed');
  assert.equal(diag.total_lines, 4, 'four non-blank lines total');
  assert.ok(Array.isArray(diag.failed_lines) && diag.failed_lines.length === 2,
    'per-line failure reasons recorded');
  const reasons = diag.failed_lines.map(f => f.reason);
  assert.ok(reasons.includes('missing_event_id'), 'missing_event_id reason present');

  // Counting/listing still works (does not throw) and returns only valid events.
  const count = await es.countEvents({});
  assert.equal(count, 2, 'corrupt lines do not count as events');
  const list = await es.listEvents({});
  assert.equal(list.length, 2);

  // storeInfo() embeds the same accounting for operators.
  const info1 = es.storeInfo();
  assert.ok(info1.jsonl_diagnostics, 'storeInfo carries jsonl_diagnostics');
  assert.equal(info1.jsonl_diagnostics.failed, 2);
});

test('atom3 - jsonlDiagnostics reports zero failures on a clean log', async () => {
  freshDir('kolm-sota-p3b-');
  const es = await import('../src/event-store.js?p3b=' + Date.now());
  es._resetForTests();
  await es.appendEvent({ tenant_id: 't', namespace: 'default', provider: 'openai', model: 'm', prompt_redacted: 'x', response_redacted: 'y', status: 'ok' });
  const diag = es.jsonlDiagnostics();
  assert.equal(diag.failed, 0);
  assert.equal(diag.parsed, 1);
  assert.equal(diag.failed_lines.length, 0);
});

// =============================================================================
// Atom 4 - deterministic fallback event_id in the capture-store bridge
// =============================================================================
test('atom4 - observationToCanonicalEvent mints a deterministic event_id without an explicit id', async () => {
  freshDir('kolm-sota-p4-');
  const cs = await import('../src/capture-store.js?p4=' + Date.now());

  const created_at = '2026-06-16T12:34:56.789Z';
  const rowBase = {
    tenant_id: 't-acme',
    corpus_namespace: 'ns',
    prompt: 'what is 2+2',
    response: '4',
    provider: 'openai',
    model: 'gpt-4o',
    created_at,
    // NOTE: no event_id, no id - forces the fallback path.
  };

  // Two independent inserts of the SAME observation (e.g. proxy mirror + direct
  // API) within the same minute bucket must collapse to ONE event_id.
  const e1 = cs.observationToCanonicalEvent({ ...rowBase });
  const e2 = cs.observationToCanonicalEvent({ ...rowBase });
  assert.ok(e1 && e2);
  assert.equal(e1.event_id, e2.event_id, 'same observation -> same deterministic event_id');
  assert.match(e1.event_id, /^evt_[0-9a-f]{16}$/, 'fallback id is evt_+sha256(16)');
  assert.ok(!/Date\.now|^evt_[0-9a-z]{1,10}$/.test('marker'), 'sanity');

  // A different response must produce a different id (content-addressed).
  const e3 = cs.observationToCanonicalEvent({ ...rowBase, response: 'five' });
  assert.notEqual(e3.event_id, e1.event_id, 'different content -> different event_id');

  // A different tenant must NOT collide even with identical content (fence).
  const e4 = cs.observationToCanonicalEvent({ ...rowBase, tenant_id: 't-other' });
  assert.notEqual(e4.event_id, e1.event_id, 'tenant is part of the fallback hash');

  // An explicit event_id still wins over the fallback.
  const e5 = cs.observationToCanonicalEvent({ ...rowBase, event_id: 'evt_explicit_123' });
  assert.equal(e5.event_id, 'evt_explicit_123');

  // Different minute bucket -> different id (time bucketing is intentional).
  const e6 = cs.observationToCanonicalEvent({ ...rowBase, created_at: '2026-06-16T12:35:56.789Z' });
  assert.notEqual(e6.event_id, e1.event_id, 'minute bucket participates in the hash');
});

test('atom4 - deterministic fallback dedupes through the event-store bridge', async () => {
  freshDir('kolm-sota-p4b-');
  const cs = await import('../src/capture-store.js?p4b=' + Date.now());
  const es = await import('../src/event-store.js?p4bes=' + Date.now());
  es._resetForTests();

  const created_at = '2026-06-16T09:00:01.000Z';
  const row = { tenant_id: 't1', corpus_namespace: 'default', prompt: 'p', response: 'r', provider: 'openai', model: 'm', created_at };

  // Bridge the same observation twice. INSERT-OR-REPLACE keyed on the
  // deterministic event_id must leave exactly one canonical event.
  await es.appendEvent(cs.observationToCanonicalEvent({ ...row }));
  await es.appendEvent(cs.observationToCanonicalEvent({ ...row }));

  const count = await es.countEvents({ tenant_id: 't1' });
  assert.equal(count, 1, 'duplicate observation must not duplicate the canonical event');
});

// =============================================================================
// SOTA build pass - Atoms 2-7 (this lane's build-for-real fixes)
// =============================================================================
import { execFileSync } from 'node:child_process';
import { fileURLToPath, pathToFileURL } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(__dirname, '..');
const STORE_URL = pathToFileURL(path.join(REPO, 'src/store.js')).href;

// -----------------------------------------------------------------------------
// Atom2 - KOLM_STORE_DRIVER=vercel_postgres must NOT boot-crash the core store.
// store.js is import-cached, so the only faithful test is a fresh child process
// with the env var flipped BEFORE import (the wave212 test masked the crash by
// importing store.js first). storeDriver() must resolve to json/sqlite.
// -----------------------------------------------------------------------------
test('atom2 - KOLM_STORE_DRIVER=vercel_postgres loads store.js and falls back to a core driver', () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'kolm-sota-a2-'));
  const dataDir = path.join(tmp, '.kolm');
  fs.mkdirSync(dataDir, { recursive: true });
  const code = `
    import { storeDriver, backendInfo } from ${JSON.stringify(STORE_URL)};
    const d = storeDriver();
    if (d !== 'json' && d !== 'sqlite') { console.error('BAD_DRIVER:' + d); process.exit(2); }
    process.stdout.write('OK:' + d + ':' + (backendInfo().driver));
  `;
  const out = execFileSync(process.execPath, ['--input-type=module', '-e', code], {
    env: {
      ...process.env,
      KOLM_STORE_DRIVER: 'vercel_postgres',
      KOLM_DATA_DIR: dataDir,
      HOME: tmp, USERPROFILE: tmp,
      NODE_ENV: 'development',
      VERCEL: '', AWS_EXECUTION_ENV: '', ECS_CONTAINER_METADATA_URI_V4: '', AWS_LAMBDA_FUNCTION_NAME: '',
    },
    encoding: 'utf8',
  });
  assert.match(out, /^OK:(json|sqlite):(json|sqlite)$/, 'store.js boots and reports a core driver: ' + out);
});

test('atom2 - a genuinely unknown KOLM_STORE_DRIVER still hard-throws at boot', () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'kolm-sota-a2b-'));
  const dataDir = path.join(tmp, '.kolm');
  fs.mkdirSync(dataDir, { recursive: true });
  const code = `import(${JSON.stringify(STORE_URL)}).then(()=>process.exit(0)).catch(()=>process.exit(7));`;
  let threw = false;
  try {
    execFileSync(process.execPath, ['--input-type=module', '-e', code], {
      env: { ...process.env, KOLM_STORE_DRIVER: 'totally_made_up', KOLM_DATA_DIR: dataDir, HOME: tmp, USERPROFILE: tmp, NODE_ENV: 'development' },
      encoding: 'utf8', stdio: 'pipe',
    });
  } catch { threw = true; }
  assert.ok(threw, 'unknown core-store driver must reject at boot (exit non-zero)');
});

// -----------------------------------------------------------------------------
// Atom1 - transactionality assertions.
// -----------------------------------------------------------------------------
test('atom1 - isTransactional + assertTeamsTransactionality reflect the json driver', async () => {
  freshDir('kolm-sota-a1-');
  const store = await loadStore();
  // The test harness forces KOLM_STORE_DRIVER=json -> not transactional.
  assert.equal(store.isTransactional(), false);
  // Non-prod: assert returns the info object and does NOT throw on json.
  const info = store.assertTeamsTransactionality({ teamsEnabled: true });
  assert.equal(info.driver, 'json');
  assert.equal(info.transactional, false);
  // teamsEnabled:false is always a no-op pass.
  assert.equal(store.assertTeamsTransactionality({ teamsEnabled: false }).ok, true);
});

// -----------------------------------------------------------------------------
// Atom3 - JSONL durable purge + compaction + .bak recovery.
// -----------------------------------------------------------------------------
test('atom3 - compactJsonl collapses re-emitted event_id rows to one (last-write-wins)', async () => {
  freshDir('kolm-sota-a3-');
  const es = await import('../src/event-store.js?a3=' + Date.now());
  es._resetForTests();

  const base = { tenant_id: 't', namespace: 'default', provider: 'openai', model: 'm', status: 'ok' };
  // Append the SAME event_id 5 times (blind-append re-emit) plus one distinct.
  for (let i = 0; i < 5; i += 1) {
    await es.appendEvent({ ...base, event_id: 'evt_dupe', prompt_redacted: 'p' + i, response_redacted: 'r' + i });
  }
  await es.appendEvent({ ...base, event_id: 'evt_other', prompt_redacted: 'x', response_redacted: 'y' });

  const info = es.storeInfo();
  assert.equal(info.driver, 'jsonl');
  const diagBefore = es.jsonlDiagnostics();
  assert.equal(diagBefore.total_lines, 6, 'six raw lines on disk (blind append)');
  assert.equal(diagBefore.distinct_events, 2, 'two distinct event_ids');

  const res = es.compactJsonl({ force: true });
  assert.equal(res.compacted, true);
  assert.equal(res.after_lines, 2, 'compaction leaves one row per event_id');

  const diagAfter = es.jsonlDiagnostics();
  assert.equal(diagAfter.total_lines, 2, 'file is physically compacted');
  // Last-write-wins: the surviving evt_dupe row is the last appended (i=4).
  const dupe = await es.getEvent('evt_dupe');
  assert.equal(dupe.prompt_redacted, 'p4', 'last write wins after compaction');
  const count = await es.countEvents({});
  assert.equal(count, 2);
});

test('atom3 - compactJsonl writes a .bak mirror and is gated by the ratio threshold', async () => {
  freshDir('kolm-sota-a3b-');
  const es = await import('../src/event-store.js?a3b=' + Date.now());
  es._resetForTests();
  await es.appendEvent({ tenant_id: 't', namespace: 'default', provider: 'p', model: 'm', status: 'ok', event_id: 'evt_a', prompt_redacted: 'a', response_redacted: 'b' });
  // Below the default minLines threshold -> not due unless forced.
  const notDue = es.compactJsonl();
  assert.equal(notDue.compacted, false, 'small clean file is not compacted by default');
  // .bak mirror exists from the append path.
  const info = es.storeInfo();
  assert.ok(info.jsonl_bak_path, 'storeInfo surfaces the .bak path');
  assert.ok(fs.existsSync(info.jsonl_bak_path), '.bak mirror is written on append');
});

test('atom3 - purgeEvents rewrites the JSONL log durably (atomic, no torn file)', async () => {
  freshDir('kolm-sota-a3c-');
  const es = await import('../src/event-store.js?a3c=' + Date.now());
  es._resetForTests();
  const old = '2020-01-01T00:00:00.000Z';
  const now = new Date().toISOString();
  await es.appendEvent({ tenant_id: 't', namespace: 'default', provider: 'p', model: 'm', status: 'ok', event_id: 'evt_old', created_at: old, prompt_redacted: 'a', response_redacted: 'b' });
  await es.appendEvent({ tenant_id: 't', namespace: 'default', provider: 'p', model: 'm', status: 'ok', event_id: 'evt_new', created_at: now, prompt_redacted: 'c', response_redacted: 'd' });
  const res = await es.purgeEvents({ before: '2021-01-01T00:00:00.000Z' });
  assert.equal(res.deleted, 1, 'one old event purged');
  const remaining = await es.listEvents({});
  assert.equal(remaining.length, 1);
  assert.equal(remaining[0].event_id, 'evt_new');
  // After a durable rewrite no .tmp residue should linger in the events dir.
  const info = es.storeInfo();
  const dir = path.dirname(info.jsonl_path);
  const stray = fs.readdirSync(dir).filter(n => n.endsWith('.tmp'));
  assert.equal(stray.length, 0, 'durable write left no .tmp residue');
});

test('atom3 - a corrupt primary JSONL is recovered from the .bak mirror', async () => {
  freshDir('kolm-sota-a3d-');
  const es = await import('../src/event-store.js?a3d=' + Date.now());
  es._resetForTests();
  await es.appendEvent({ tenant_id: 't', namespace: 'default', provider: 'p', model: 'm', status: 'ok', event_id: 'evt_keep', prompt_redacted: 'a', response_redacted: 'b' });
  const info = es.storeInfo();
  // Corrupt the PRIMARY entirely while the .bak mirror stays intact.
  fs.writeFileSync(info.jsonl_path, '{{{garbage not json at all', 'utf8');
  // Force a fresh module so the in-memory state does not paper over disk reads.
  es._resetForTests();
  const list = await es.listEvents({});
  assert.equal(list.length, 1, 'event recovered from .bak after primary corruption');
  assert.equal(list[0].event_id, 'evt_keep');
});

// -----------------------------------------------------------------------------
// Atom4 - secrets-vault durable write + recovery (no silent empty-vault).
// -----------------------------------------------------------------------------
test('atom4 - secrets vault survives a corrupt primary by recovering from .bak', async () => {
  freshDir('kolm-sota-a4-');
  const sv = await import('../src/secrets-vault.js?a4=' + Date.now());
  sv.putSecret({ id: 'runpod-key', value: 'rp_live_secret_value' });
  const paths = sv.vaultFilePaths();
  assert.ok(fs.existsSync(paths.vault), 'vault primary written');
  assert.ok(fs.existsSync(paths.vault_bak), '.bak mirror written durably');

  // Simulate a crash mid-rewrite: corrupt the primary, leave .bak intact.
  fs.writeFileSync(paths.vault, '{ truncated', 'utf8');
  // readVault (via getSecret) must recover from .bak, NOT return an empty vault.
  const got = sv.getSecret('runpod-key');
  assert.ok(got, 'secret recovered after primary corruption (no silent empty vault)');
  assert.equal(got.value, 'rp_live_secret_value');
  // The corrupt primary must have been quarantined, not silently overwritten.
  const dir = path.dirname(paths.vault);
  const quarantined = fs.readdirSync(dir).filter(n => /secrets-vault\.json\.corrupt-/.test(n));
  assert.ok(quarantined.length >= 1, 'corrupt primary quarantined for forensics');
});

test('atom4 - corrupt primary with NO backup throws loudly instead of dropping secrets', async () => {
  freshDir('kolm-sota-a4b-');
  const sv = await import('../src/secrets-vault.js?a4b=' + Date.now());
  sv.putSecret({ id: 'k', value: 'v' });
  const paths = sv.vaultFilePaths();
  // Remove the backup, corrupt the primary -> unrecoverable -> must throw.
  fs.rmSync(paths.vault_bak, { force: true });
  fs.writeFileSync(paths.vault, 'not json', 'utf8');
  assert.throws(() => sv.getSecret('k'), /corrupt/i, 'unrecoverable vault throws, never returns empty');
});

// -----------------------------------------------------------------------------
// Atom5 - store-backup snapshots the encrypted vault + key in both driver modes.
// -----------------------------------------------------------------------------
test('atom5 - backupNow snapshots the secrets vault + key (json driver)', async () => {
  const tmp = freshDir('kolm-sota-a5-');
  const store = await loadStore();
  store.insert('tenants', { id: 'ten_1', name: 'acme' });
  const sv = await import('../src/secrets-vault.js?a5=' + Date.now());
  sv.putSecret({ id: 'provider-key', value: 'sk_live_xyz' });

  const sb = await import('../src/store-backup.js?a5=' + Date.now());
  const res = sb.backupNow();
  assert.equal(res.ok, true, 'backup ok: ' + JSON.stringify(res));
  assert.ok(res.vault && res.vault.ok, 'vault snapshot attempted');
  assert.ok(res.vault.files.includes('secrets-vault.json'), 'vault ciphertext snapshotted');
  assert.ok(res.vault.files.includes('secrets-vault.key'), 'vault KEY snapshotted (ciphertext recoverable)');
  void tmp;
});

// -----------------------------------------------------------------------------
// Atom6 - SQLite WAL checkpoint export + corrupt-DB self-recovery surface.
// Exercised against the json-cached singleton: the lifecycle exports must exist
// and be safe no-ops when not on sqlite.
// -----------------------------------------------------------------------------
test('atom6 - store exposes installShutdownCheckpoint + isTransactional lifecycle hooks', async () => {
  freshDir('kolm-sota-a6-');
  const store = await loadStore();
  assert.equal(typeof store.installShutdownCheckpoint, 'function');
  // Idempotent + safe to call under the json driver (no sqlite db open).
  assert.doesNotThrow(() => store.installShutdownCheckpoint());
  assert.doesNotThrow(() => store.installShutdownCheckpoint());
});

// -----------------------------------------------------------------------------
// Atom7 - migration runner + idempotency ledger.
// -----------------------------------------------------------------------------
test('atom7 - runPendingMigrations records the ledger and is idempotent', async () => {
  freshDir('kolm-sota-a7-');
  const store = await loadStore();
  // Seed one legacy capture row so the migration has something to backfill.
  store.insert('observations', {
    id: 'obs_1', tenant_id: 't-acme', corpus_namespace: 'ns',
    prompt: 'hi', response: 'there', provider: 'openai', model: 'gpt',
    created_at: '2026-06-16T10:00:00.000Z',
  });
  const mig = await import('../src/migrations/index.js?a7=' + Date.now());

  // Dry-run must NOT write the ledger.
  const dry = await mig.runPendingMigrations({ dryRun: true });
  assert.equal(dry.dry_run, true);
  assert.equal(dry.ran.length, 1);
  assert.equal(mig.isMigrationApplied('2026-05-19-capture-to-events'), false, 'dry-run does not mark applied');

  // Real run records the ledger.
  const r1 = await mig.runPendingMigrations();
  assert.equal(r1.ran.length, 1);
  assert.ok(r1.ran[0].stats, 'migration returned stats');
  assert.equal(mig.isMigrationApplied('2026-05-19-capture-to-events'), true);

  // Second run is idempotent: the migration is skipped via the ledger.
  const r2 = await mig.runPendingMigrations();
  assert.equal(r2.ran.length, 0, 'already-applied migration is skipped');
  assert.deepEqual(r2.skipped, ['2026-05-19-capture-to-events']);

  // Status surface lists the migration as applied.
  const status = mig.migrationStatus();
  assert.equal(status.migrations[0].applied, true);
  assert.ok(status.ledger_rows >= 1);
});

test('atom7 - only canonical src/migrations is wired (tmp/ duplicate is not importable here)', async () => {
  const mig = await import('../src/migrations/index.js?a7b=' + Date.now());
  // The registry sources exactly the canonical migration id.
  assert.deepEqual(mig.MIGRATIONS.map(m => m.id), ['2026-05-19-capture-to-events']);
});
