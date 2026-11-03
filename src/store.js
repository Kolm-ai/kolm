// Persistent store facade.
//
// Development defaults to durable JSON files for zero-dependency local use.
// Production can opt into the SQLite backend with KOLM_STORE_DRIVER=sqlite,
// giving us transactional writes and a real database file without adding a
// runtime package dependency on modern Node builds.

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import { createRequire } from 'node:module';
import { isProductionRuntime } from './env.js';

const require = createRequire(import.meta.url);

const ON_VERCEL = !!(process.env.VERCEL || process.env.AWS_LAMBDA_FUNCTION_NAME);
const BUNDLED_DATA_DIR = path.resolve('data');
const PREFERRED_DATA_DIR = process.env.KOLM_DATA_DIR
  ? path.resolve(process.env.KOLM_DATA_DIR)
  : (ON_VERCEL ? '/tmp/data' : BUNDLED_DATA_DIR);

function probeWritable(dir) {
  // Verify we can actually create/write files in this directory. On network
  // persistent volumes the directory can exist but be owned by a previous
  // container's user → EACCES on every write. When that happens we must
  // degrade to a writable fallback OR the boot path can succeed but every
  // write-bound HTTP route (signup, capture, deploy) will 500 forever.
  try {
    fs.mkdirSync(dir, { recursive: true });
    const marker = path.join(dir, '.kolm-write-probe');
    fs.writeFileSync(marker, String(Date.now()));
    fs.unlinkSync(marker);
    return true;
  } catch {
    return false;
  }
}

let DATA_DIR = PREFERRED_DATA_DIR;
let STORE_EPHEMERAL = false;
if (!probeWritable(DATA_DIR)) {
  const fallback = path.join(os.tmpdir(), 'kolm-data');
  // A paid product must NEVER silently run on ephemeral /tmp (audit + billing
  // state would vanish on the next restart with no error). In a production-like
  // environment, fail boot LOUDLY instead of degrading invisibly, unless the
  // operator explicitly opts in with KOLM_ALLOW_EPHEMERAL=1.
  const productionLike = isProductionRuntime();
  if (productionLike && process.env.KOLM_ALLOW_EPHEMERAL !== '1') {
    throw new Error(`[store] FATAL: KOLM_DATA_DIR ${DATA_DIR} is not writable in a production environment. Refusing to fall back to ephemeral /tmp because paid + audit state would be silently lost on restart. Fix the volume mount/permissions, or set KOLM_ALLOW_EPHEMERAL=1 to explicitly accept ephemeral storage.`);
  }
  if (probeWritable(fallback)) {
    console.error(`[store] WARNING: KOLM_DATA_DIR ${DATA_DIR} is not writable (EACCES or similar). Falling back to ${fallback}. State written here will NOT survive container restarts - fix volume permissions to recover persistence.`);
    DATA_DIR = fallback;
    STORE_EPHEMERAL = true;
  } else {
    console.error(`[store] FATAL: neither ${DATA_DIR} nor ${fallback} is writable. Persistent state operations will throw at write time.`);
  }
}

function detectDefaultDriver() {
  // Production-like environments default to SQLite when the runtime supports
  // it; this gives us transactional writes without callers opting in. Local
  // development keeps the dependency-free JSON files unless explicitly asked.
  const productionLike = isProductionRuntime();
  if (!productionLike) return 'json';
  try {
    require('node:sqlite');
    return 'sqlite';
  } catch {
    return 'json';
  }
}

// Atom2 - KOLM_STORE_DRIVER is a SHARED env var across two namespaces:
//   - the core synchronous row store (this module): json | sqlite
//   - the async, pluggable capture/durable driver (src/capture-store.js):
//     vercel_postgres | vercel_kv
// Both are documented valid values (docs/durability.md, the store-drivers/*.js
// headers, capture-store.js:44-45). Before this fix, setting
// KOLM_STORE_DRIVER=vercel_postgres in a real deploy boot-CRASHED the whole
// server here (the core store rejected the same env var the capture path
// requires) before any route loaded. The capture-store consumes the pluggable
// value; the core store must treat it as "not one of MY drivers" and fall back
// to the detected json/sqlite core driver instead of throwing. We keep the hard
// throw ONLY for genuinely unknown strings.
const CAPTURE_ONLY_DRIVERS = new Set(['vercel_postgres', 'vercel_kv']);
const RAW_STORE_DRIVER = (process.env.KOLM_STORE_DRIVER || '').toLowerCase();
let STORE_DRIVER;
if (!RAW_STORE_DRIVER) {
  STORE_DRIVER = detectDefaultDriver();
} else if (CAPTURE_ONLY_DRIVERS.has(RAW_STORE_DRIVER)) {
  // The operator selected a pluggable capture/durable driver. That driver is
  // owned by capture-store.js; the core row store keeps running on its own
  // detected json/sqlite driver. Emit a one-line notice (not an error) so the
  // operator understands which subsystem the env var actually steers.
  STORE_DRIVER = detectDefaultDriver();
  console.error(`[store] notice: KOLM_STORE_DRIVER=${RAW_STORE_DRIVER} selects the pluggable capture driver (src/capture-store.js); the core row store runs on "${STORE_DRIVER}".`);
} else if (['json', 'jsonl', 'sqlite'].includes(RAW_STORE_DRIVER)) {
  // 'jsonl' is a historical alias for the dependency-free JSON file store
  // (JSON-Lines-era naming). It maps to the 'json' core driver. Kept as a valid
  // value so the W381/W397/W409/W411 corpus + train/holdout suites - which select
  // it explicitly to force the file store over sqlite - run against a real backend
  // instead of boot-throwing on a fresh module load.
  STORE_DRIVER = RAW_STORE_DRIVER === 'jsonl' ? 'json' : RAW_STORE_DRIVER;
} else {
  throw new Error(`Unsupported KOLM_STORE_DRIVER "${RAW_STORE_DRIVER}". Core-store values: "json" or "sqlite". Pluggable capture-driver values: "vercel_postgres" or "vercel_kv".`);
}

const SQLITE_PATH = process.env.KOLM_DB_PATH
  ? path.resolve(process.env.KOLM_DB_PATH)
  : path.join(DATA_DIR, 'kolm.sqlite');

// Fail-closed: never silently run a production-like instance on the JSON store.
// JSON tables are per-process flat files - not durable/consistent across
// multi-node or restart, so a prod deploy that falls back to json would lose
// billing + audit state. Mirrors the env.js readiness stance. Opt out only with
// an explicit KOLM_ALLOW_JSON_STORE=true.
{
  const _prodLike = isProductionRuntime();
  if (_prodLike && STORE_DRIVER === 'json' && process.env.KOLM_ALLOW_JSON_STORE !== 'true') {
    throw new Error('[store] FATAL: KOLM_STORE_DRIVER=json (or defaulted) in a production environment. JSON files are per-process and not durable for restart/multi-node state. Set KOLM_STORE_DRIVER=sqlite (recommended), or KOLM_ALLOW_JSON_STORE=true to explicitly accept it.');
  }
}

fs.mkdirSync(DATA_DIR, { recursive: true });
fs.mkdirSync(path.join(DATA_DIR, 'cache'), { recursive: true });
if (STORE_DRIVER === 'sqlite') fs.mkdirSync(path.dirname(SQLITE_PATH), { recursive: true });

if (ON_VERCEL && fs.existsSync(BUNDLED_DATA_DIR) && BUNDLED_DATA_DIR !== DATA_DIR) {
  for (const f of fs.readdirSync(BUNDLED_DATA_DIR)) {
    const src = path.join(BUNDLED_DATA_DIR, f);
    const dst = path.join(DATA_DIR, f);
    if (fs.existsSync(dst)) continue;
    try {
      const stat = fs.statSync(src);
      if (stat.isFile()) fs.copyFileSync(src, dst);
    } catch { // deliberate: cleanup
      // Best-effort seed; safe to skip.
    }
  }
}

const jsonTables = new Map();
const sqliteTables = new Set();
let sqliteDb = null;

function tablePath(name) {
  return path.join(DATA_DIR, `${name}.json`);
}

function backupPath(name) {
  return tablePath(name) + '.bak';
}

function fsyncDir(dir) {
  try {
    const fd = fs.openSync(dir, 'r');
    try { fs.fsyncSync(fd); } finally { fs.closeSync(fd); }
  } catch { // deliberate: cleanup
    // Directory fsync is best-effort and not uniformly supported on Windows.
  }
}

function sleepSync(ms) {
  try {
    Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms);
  } catch {
    const until = Date.now() + ms;
    while (Date.now() < until) {}
  }
}

function replaceFileWithRetry(tmp, file) {
  let lastErr = null;
  for (let attempt = 0; attempt < 8; attempt += 1) {
    try {
      fs.renameSync(tmp, file);
      return;
    } catch (err) {
      lastErr = err;
      if (!['EPERM', 'EACCES', 'EBUSY'].includes(err && err.code)) throw err;
      sleepSync(10 * (2 ** attempt));
    }
  }
  // Windows can briefly hold the destination open during antivirus/indexing or
  // adjacent test-server reads. Preserve the write instead of dropping the row:
  // copy over the destination after retries, fsync below, then remove temp.
  try {
    fs.copyFileSync(tmp, file);
    try { fs.rmSync(tmp, { force: true }); } catch {} // deliberate: cleanup
    return;
  } catch {
    throw lastErr;
  }
}

function writeFileDurably(file, text) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const tmp = `${file}.${process.pid}.${Date.now()}.${crypto.randomBytes(4).toString('hex')}.tmp`;
  let fd = null;
  try {
    fd = fs.openSync(tmp, 'w', 0o600);
    fs.writeFileSync(fd, text, 'utf8');
    fs.fsyncSync(fd);
    fs.closeSync(fd);
    fd = null;
    replaceFileWithRetry(tmp, file);
    fsyncDir(path.dirname(file));
  } catch (err) {
    if (fd !== null) {
      try { fs.closeSync(fd); } catch {} // deliberate: cleanup
    }
    try { fs.rmSync(tmp, { force: true }); } catch {} // deliberate: cleanup
    throw err;
  }
}

function assertRowsArray(name, rows) {
  if (!Array.isArray(rows)) {
    throw new Error(`Store table "${name}" must be a JSON array`);
  }
}

function readRowsFile(name, file) {
  const parsed = JSON.parse(fs.readFileSync(file, 'utf8'));
  assertRowsArray(name, parsed);
  return parsed;
}

function quarantineCorruptFile(file) {
  if (!fs.existsSync(file)) return null;
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  const dest = `${file}.corrupt-${stamp}-${process.pid}-${crypto.randomBytes(3).toString('hex')}`;
  try {
    fs.renameSync(file, dest);
    return dest;
  } catch {
    try {
      fs.copyFileSync(file, dest);
      return dest;
    } catch {
      return null;
    }
  }
}

function loadJsonTable(name) {
  if (jsonTables.has(name)) return jsonTables.get(name);
  const p = tablePath(name);
  let rows = [];
  if (fs.existsSync(p)) {
    try {
      rows = readRowsFile(name, p);
    } catch (primaryErr) {
      const corruptPrimary = quarantineCorruptFile(p);
      const bak = backupPath(name);
      if (fs.existsSync(bak)) {
        try {
          rows = readRowsFile(name, bak);
          try {
            writeFileDurably(p, JSON.stringify(rows, null, 2));
            console.error(`[store] recovered ${name}.json from backup after read failure`);
          } catch (writeErr) {
            console.error(`[store] recovered ${name}.json from backup but flush to primary failed: ${writeErr.message}`);
          }
        } catch (backupErr) {
          const corruptBackup = quarantineCorruptFile(bak);
          if (process.env.KOLM_STORE_STRICT === '1') {
            throw new Error(
              `Cannot load store table "${name}": primary JSON is invalid` +
              `${corruptPrimary ? `; quarantined at ${corruptPrimary}` : ''}; ` +
              `backup is invalid${corruptBackup ? `; quarantined at ${corruptBackup}` : ''}: ${backupErr.message}`,
            );
          }
          console.error(
            `[store] FATAL recoverable: ${name}.json + ${name}.json.bak both unreadable` +
            `${corruptPrimary ? ` (primary quarantined at ${corruptPrimary})` : ''}` +
            `${corruptBackup ? ` (backup quarantined at ${corruptBackup})` : ''}` +
            `; starting with empty table. primary: ${primaryErr.message}; backup: ${backupErr.message}`,
          );
          rows = [];
        }
      } else {
        if (process.env.KOLM_STORE_STRICT === '1') {
          throw new Error(
            `Cannot load store table "${name}": primary JSON is invalid` +
            `${corruptPrimary ? `; quarantined at ${corruptPrimary}` : ''}: ${primaryErr.message}`,
          );
        }
        console.error(
          `[store] FATAL recoverable: ${name}.json unreadable and no backup present` +
          `${corruptPrimary ? ` (primary quarantined at ${corruptPrimary})` : ''}` +
          `; starting with empty table. primary: ${primaryErr.message}`,
        );
        rows = [];
      }
    }
  }
  jsonTables.set(name, rows);
  return rows;
}

function flushJsonTable(name) {
  const rows = jsonTables.get(name) || [];
  assertRowsArray(name, rows);
  const p = tablePath(name);
  const json = JSON.stringify(rows, null, 2);
  writeFileDurably(p, json);
  try {
    writeFileDurably(backupPath(name), json);
  } catch (err) {
    console.error(`[store] warning: failed to refresh backup for ${name}.json: ${err.message}`);
  }
}

// Atom6 (2) - quarantine a corrupt SQLite main DB and try to recover from the
// newest store-backup snapshot, giving the production SQLite driver the same
// fail-soft posture the JSON driver already has (.bak/quarantine). Returns true
// if a usable DB file is in place at SQLITE_PATH afterwards, false otherwise.
// Never throws across its boundary; logs loudly.
function quarantineAndRecoverSqlite(reason) {
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  const corruptDest = `${SQLITE_PATH}.corrupt-${stamp}-${process.pid}`;
  // Move the corrupt primary + its WAL/SHM sidecars aside so a fresh open does
  // not replay a poisoned WAL on top of any restored file.
  for (const suffix of ['', '-wal', '-shm']) {
    const f = SQLITE_PATH + suffix;
    try { if (fs.existsSync(f)) fs.renameSync(f, corruptDest + suffix); } catch { /* best-effort */ }
  }
  console.error(`[store] FATAL recoverable: kolm.sqlite ${reason}; quarantined at ${corruptDest}. Attempting restore from newest store-backup snapshot.`);
  // Find the newest VACUUM-INTO snapshot under DATA_DIR/backups and copy it in.
  try {
    const backupsDir = path.join(DATA_DIR, 'backups');
    if (fs.existsSync(backupsDir)) {
      const snaps = fs.readdirSync(backupsDir)
        .filter(n => n.startsWith('kolm-') && n.endsWith('.sqlite'))
        .sort(); // ISO timestamps sort chronologically
      const newest = snaps[snaps.length - 1];
      if (newest) {
        const src = path.join(backupsDir, newest);
        fs.copyFileSync(src, SQLITE_PATH);
        console.error(`[store] recovered kolm.sqlite from backup snapshot ${newest}`);
        return true;
      }
    }
  } catch (e) {
    console.error(`[store] backup restore failed after quarantine: ${e.message}; starting with a fresh empty DB.`);
  }
  // No snapshot available: a fresh DatabaseSync open will create an empty DB at
  // SQLITE_PATH. Seed re-import from the *.json table files still applies.
  return false;
}

function getSqliteDb() {
  if (sqliteDb) return sqliteDb;
  let DatabaseSync;
  try {
    ({ DatabaseSync } = require('node:sqlite'));
  } catch (err) {
    throw new Error(`KOLM_STORE_DRIVER=sqlite requires Node with node:sqlite support: ${err.message}`);
  }
  try {
    sqliteDb = new DatabaseSync(SQLITE_PATH);
  } catch (openErr) {
    // Atom6 - a corrupt/unopenable kolm.sqlite must not hard-throw at first
    // query with no degrade path. Quarantine + restore + retry once.
    sqliteDb = null;
    quarantineAndRecoverSqlite(`failed to open (${openErr.message})`);
    sqliteDb = new DatabaseSync(SQLITE_PATH);
  }
  // Atom6 - integrity gate. A silently-corrupt page set would otherwise throw
  // mid-transaction much later. Check once at open; on failure quarantine the
  // file, restore the newest snapshot, and re-open.
  try {
    const ic = sqliteDb.prepare('PRAGMA integrity_check').get();
    const result = ic && (ic.integrity_check || ic['integrity_check'] || Object.values(ic)[0]);
    if (result && String(result).toLowerCase() !== 'ok') {
      try { sqliteDb.close(); } catch { /* */ }
      sqliteDb = null;
      quarantineAndRecoverSqlite(`failed PRAGMA integrity_check (${String(result).slice(0, 120)})`);
      sqliteDb = new DatabaseSync(SQLITE_PATH);
    }
  } catch (icErr) {
    // integrity_check itself threw -> treat the file as corrupt.
    try { if (sqliteDb) sqliteDb.close(); } catch { /* */ }
    sqliteDb = null;
    quarantineAndRecoverSqlite(`integrity_check threw (${icErr.message})`);
    sqliteDb = new DatabaseSync(SQLITE_PATH);
  }
  // WAL needs an mmap'd shared-memory index, which SQLite only supports on a
  // local disk. On a network volume (EFS/NFS) use a rollback journal instead:
  // the ECS task definition sets KOLM_SQLITE_JOURNAL_MODE=TRUNCATE
  // (infra/aws/kolm-api.yaml). store-backup.js opens a second connection for
  // VACUUM INTO, so exclusive locking is not an option here.
  const journalModeEnv = String(process.env.KOLM_SQLITE_JOURNAL_MODE || '').toUpperCase();
  const journalMode = /^(WAL|DELETE|TRUNCATE|PERSIST)$/.test(journalModeEnv) ? journalModeEnv : 'WAL';
  sqliteDb.exec(`
    PRAGMA journal_mode = ${journalMode};
    PRAGMA synchronous = FULL;
    PRAGMA foreign_keys = ON;
    PRAGMA busy_timeout = 30000;
    CREATE TABLE IF NOT EXISTS kolm_store_rows (
      row_id INTEGER PRIMARY KEY AUTOINCREMENT,
      table_name TEXT NOT NULL,
      json TEXT NOT NULL,
      created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
      updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_kolm_store_rows_table_row_id
      ON kolm_store_rows(table_name, row_id);
    -- Expression indexes on the hot findByField() lookups (tenant fences, slug
    -- resolution, Stripe webhook + Continuous tick). findByField() filters on
    -- json_extract(json, '$.<field>') = ?, which these indexes serve directly,
    -- so webhook/tick/trust stay fast as agent_audits + asr_subscriptions grow.
    CREATE INDEX IF NOT EXISTS idx_kolm_store_rows_tenant
      ON kolm_store_rows(table_name, json_extract(json, '$.tenant_id'));
    CREATE INDEX IF NOT EXISTS idx_kolm_store_rows_rowid_field
      ON kolm_store_rows(table_name, json_extract(json, '$.id'));
    CREATE INDEX IF NOT EXISTS idx_kolm_store_rows_public_slug
      ON kolm_store_rows(table_name, json_extract(json, '$.public_slug'));
    CREATE INDEX IF NOT EXISTS idx_kolm_store_rows_stripe_sub
      ON kolm_store_rows(table_name, json_extract(json, '$.stripe_subscription_id'));
    CREATE TABLE IF NOT EXISTS kolm_store_meta (
      table_name TEXT PRIMARY KEY,
      imported_at TEXT NOT NULL
    );
  `);
  return sqliteDb;
}

function sqliteTableImported(db, name) {
  return !!db.prepare('SELECT 1 AS ok FROM kolm_store_meta WHERE table_name = ?').get(name);
}

function markSqliteTableImported(db, name) {
  db.prepare('INSERT OR IGNORE INTO kolm_store_meta (table_name, imported_at) VALUES (?, ?)').run(name, new Date().toISOString());
}

function importJsonSeedIntoSqlite(db, name) {
  if (sqliteTableImported(db, name)) return;
  const p = tablePath(name);
  const rows = fs.existsSync(p) ? readRowsFile(name, p) : [];
  runInTxn(db, () => {
    const insertStmt = db.prepare('INSERT INTO kolm_store_rows (table_name, json) VALUES (?, ?)');
    for (const row of rows) insertStmt.run(name, JSON.stringify(row));
    markSqliteTableImported(db, name);
  });
}

// Re-entrant transaction helper used by every multi-statement write below.
// If we are already inside a withTransaction (BEGIN IMMEDIATE) or another
// runInTxn (SAVEPOINT), reuse the existing scope via SAVEPOINT so SQLite
// does not throw "cannot start a transaction within a transaction".
let _txnDepth = 0;
let _savepointSeq = 0;
function runInTxn(db, fn) {
  if (_txnDepth === 0) {
    db.exec('BEGIN IMMEDIATE');
    _txnDepth = 1;
    try { fn(); db.exec('COMMIT'); _txnDepth = 0; }
    catch (e) { try { db.exec('ROLLBACK'); } catch {} _txnDepth = 0; throw e; } // deliberate: cleanup
    return;
  }
  const sp = `sp_${++_savepointSeq}`;
  db.exec(`SAVEPOINT ${sp}`);
  _txnDepth++;
  try { fn(); db.exec(`RELEASE ${sp}`); _txnDepth--; }
  catch (e) { try { db.exec(`ROLLBACK TO ${sp}`); db.exec(`RELEASE ${sp}`); } catch {} _txnDepth--; throw e; } // deliberate: cleanup
}

function sqliteRows(name) {
  const db = getSqliteDb();
  sqliteTables.add(name);
  importJsonSeedIntoSqlite(db, name);
  return db
    .prepare('SELECT row_id AS rowid, json FROM kolm_store_rows WHERE table_name = ? ORDER BY row_id')
    .all(name)
    .map(row => ({ rowid: row.rowid, value: JSON.parse(row.json) }));
}

function sqliteAll(name) {
  return sqliteRows(name).map(row => row.value);
}

function sqliteInsert(table, row) {
  const db = getSqliteDb();
  sqliteTables.add(table);
  importJsonSeedIntoSqlite(db, table);
  db.prepare('INSERT INTO kolm_store_rows (table_name, json) VALUES (?, ?)').run(table, JSON.stringify(row));
  return row;
}

function sqliteUpdate(table, predicate, patch) {
  const db = getSqliteDb();
  const rows = sqliteRows(table);
  const updateStmt = db.prepare('UPDATE kolm_store_rows SET json = ?, updated_at = CURRENT_TIMESTAMP WHERE rowid = ?');
  let n = 0;
  runInTxn(db, () => {
    for (const row of rows) {
      if (!predicate(row.value)) continue;
      Object.assign(row.value, patch);
      updateStmt.run(JSON.stringify(row.value), row.rowid);
      n++;
    }
  });
  return n;
}

function sqliteRemove(table, predicate) {
  const db = getSqliteDb();
  const rows = sqliteRows(table);
  const deleteStmt = db.prepare('DELETE FROM kolm_store_rows WHERE rowid = ?');
  let n = 0;
  runInTxn(db, () => {
    for (const row of rows) {
      if (!predicate(row.value)) continue;
      deleteStmt.run(row.rowid);
      n++;
    }
  });
  return n;
}

function sqliteReset() {
  const db = getSqliteDb();
  const names = [...sqliteTables];
  const deleteStmt = db.prepare('DELETE FROM kolm_store_rows WHERE table_name = ?');
  runInTxn(db, () => {
    for (const name of names) {
      deleteStmt.run(name);
      markSqliteTableImported(db, name);
    }
  });
}

export function id(prefix = 'id') {
  const r = crypto.randomBytes(8).toString('hex');
  return `${prefix}_${Date.now().toString(36)}${r}`;
}

export function insert(table, row) {
  if (STORE_DRIVER === 'sqlite') return sqliteInsert(table, row);
  const rows = loadJsonTable(table);
  rows.push(row);
  flushJsonTable(table);
  return row;
}

export function update(table, predicate, patch) {
  if (STORE_DRIVER === 'sqlite') return sqliteUpdate(table, predicate, patch);
  const rows = loadJsonTable(table);
  let n = 0;
  for (const row of rows) {
    if (predicate(row)) { Object.assign(row, patch); n++; }
  }
  flushJsonTable(table);
  return n;
}

export function find(table, predicate = () => true) {
  return all(table).filter(predicate);
}

export function findOne(table, predicate) {
  return all(table).find(predicate) || null;
}

// Indexed equality lookup primitive. In sqlite mode this issues a
// `WHERE json_extract(json, '$.<field>') = ?` query (SQLite >= 3.38 ships
// JSON1 by default) which becomes an indexed scan once an expression index
// exists on the field. In json mode it falls back to a full table scan.
// Use this in place of `all(t).filter(r => r[field] === value)` for any
// tenant- or hash-scoped lookup that previously read the entire table into
// memory.
export function findByField(table, field, value) {
  if (typeof field !== 'string' || field.length === 0) {
    throw new Error('findByField(table, field, value): field must be a non-empty string');
  }
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(field)) {
    throw new Error(`findByField(table, field, value): unsafe field name ${JSON.stringify(field)}`);
  }
  if (STORE_DRIVER === 'sqlite') {
    const db = getSqliteDb();
    sqliteTables.add(table);
    importJsonSeedIntoSqlite(db, table);
    const rows = db
      .prepare(`SELECT json FROM kolm_store_rows WHERE table_name = ? AND json_extract(json, '$.${field}') = ? ORDER BY row_id`)
      .all(table, value);
    return rows.map(r => JSON.parse(r.json));
  }
  return loadJsonTable(table).filter(r => r[field] === value);
}

// Tenant-scoped lookup convenience. Most multi-tenant tables (observations,
// invocations, audit_log, concepts, jobs) carry a `tenant` column; this
// wrapper centralises the access pattern so future SQLite expression indexes
// can be added in one place. Returns [] for missing tenants.
// W888-L: also union rows that carry a `tenant_id` column (newer fixtures and
// the api_keys-style join shape both emit tenant_id, not tenant). If a row
// matches either column we count it once.
export function findByTenant(table, tenant) {
  if (!tenant) return [];
  const a = findByField(table, 'tenant', tenant);
  const b = findByField(table, 'tenant_id', tenant);
  if (!b.length) return a;
  if (!a.length) return b;
  const seen = new Set(a.map(r => r && r.id).filter(Boolean));
  const merged = a.slice();
  for (const row of b) {
    if (!row || !row.id || seen.has(row.id)) continue;
    seen.add(row.id);
    merged.push(row);
  }
  return merged;
}

export function remove(table, predicate) {
  if (STORE_DRIVER === 'sqlite') return sqliteRemove(table, predicate);
  const rows = loadJsonTable(table);
  const kept = rows.filter(r => !predicate(r));
  jsonTables.set(table, kept);
  flushJsonTable(table);
  return rows.length - kept.length;
}

export function all(table) {
  if (STORE_DRIVER === 'sqlite') return sqliteAll(table);
  return loadJsonTable(table);
}

export function reset() {
  if (STORE_DRIVER === 'sqlite') {
    sqliteReset();
    return;
  }
  for (const t of jsonTables.keys()) {
    jsonTables.set(t, []);
    flushJsonTable(t);
  }
}

export function stats() {
  const out = {};
  for (const t of ['concepts', 'versions', 'syntheses', 'invocations', 'tenants']) {
    out[t] = all(t).length;
  }
  return out;
}

export function backendInfo() {
  return {
    driver: STORE_DRIVER,
    data_dir: DATA_DIR,
    db_path: STORE_DRIVER === 'sqlite' ? SQLITE_PATH : null,
  };
}

export function close() {
  if (sqliteDb) {
    // Atom6 (1) - checkpoint + truncate the WAL before closing so shutdown
    // leaves a fully-checkpointed main DB and small (ideally zero-byte) -wal/
    // -shm sidecars. Without this the on-disk main DB lags committed state until
    // SQLite's own auto-checkpoint runs, and operators had to delete sidecars by
    // hand during a restore (durability.md). Best-effort: never let a checkpoint
    // failure block a graceful close.
    try { sqliteDb.exec('PRAGMA wal_checkpoint(TRUNCATE)'); } catch { /* best-effort */ }
    try { sqliteDb.close(); } catch { /* best-effort */ }
    sqliteDb = null;
  }
  jsonTables.clear();
  sqliteTables.clear();
}

// Atom6 (1) - checkpoint the WAL on graceful shutdown even when the caller does
// not explicitly close the store. Registered once per process. The handler does
// NOT call process.exit() (other shutdown hooks own that); it only flushes the
// WAL into the main DB so a snapshot/restart sees committed state on disk.
let _shutdownHookInstalled = false;
export function installShutdownCheckpoint() {
  if (_shutdownHookInstalled) return;
  _shutdownHookInstalled = true;
  const checkpoint = () => {
    try {
      if (sqliteDb) sqliteDb.exec('PRAGMA wal_checkpoint(TRUNCATE)');
    } catch { /* best-effort */ }
  };
  // 'exit' is synchronous - safe for the synchronous node:sqlite PRAGMA.
  process.once('exit', checkpoint);
  // SIGTERM/SIGINT: checkpoint, then re-emit default behavior is left to other
  // handlers; we do not exit here so we don't pre-empt graceful drain.
  process.once('SIGTERM', checkpoint);
  process.once('SIGINT', checkpoint);
}
// Install automatically so the checkpoint runs even if cli/kolm.js does not
// wire an explicit close() into its shutdown path. Guarded + idempotent.
if (STORE_DRIVER === 'sqlite') {
  try { installShutdownCheckpoint(); } catch { /* best-effort */ }
}

// Wrap `fn` in a single transactional unit. In sqlite mode this is BEGIN
// IMMEDIATE / COMMIT (auto-rollback on throw); in json mode the writes are
// serial inside a single tick anyway, so the wrapper is a pass-through. Use
// this for any multi-step write that must not interleave with a concurrent
// request (entitlement charge + audit append, recipe publish + concept patch,
// team invite accept + seat allocation).
//
// Reentrancy: nested withTransaction calls reuse the outer transaction via
// SAVEPOINT. This lets a route handler open a transaction (e.g. chargeUsage)
// and call into a helper that also wraps its work (e.g. tryAppendAudit)
// without tripping "cannot start a transaction within a transaction".
export function withTransaction(fn) {
  if (STORE_DRIVER !== 'sqlite') return fn();
  const db = getSqliteDb();
  let out;
  runInTxn(db, () => {
    out = fn();
    if (out && typeof out.then === 'function') {
      throw new Error('withTransaction(fn) requires a synchronous fn; async work must run before/after the transaction');
    }
  });
  return out;
}

export function storeDriver() {
  return STORE_DRIVER;
}

// Atom1 - the Teams seat-accounting TOCTOU-safety narrative (teams.js
// removeMember/inviteToTeam/acceptInvite use withTransaction = BEGIN IMMEDIATE
// for serialized seat writes) ONLY holds on the sqlite driver. In json mode
// withTransaction is a synchronous pass-through (writes are serial within one
// tick, which is correct for a single in-process writer but NOT a real
// serialization barrier). This returns true when the active core driver gives
// withTransaction real transactional isolation. Teams + billing code can assert
// on it; capture-only pluggable drivers never steer the core store, so this is
// purely a function of json|sqlite.
export function isTransactional() {
  return STORE_DRIVER === 'sqlite';
}

// Atom1 - assert at boot that a production teams-enabled deployment runs on a
// transactional core driver (sqlite today; the future pg txn path would also
// qualify). Called by the Teams/Gateway boot path. Fails LOUD with an
// actionable hint rather than silently letting seat billing run without the
// serialization guarantee it depends on. Returns { ok, driver, transactional }
// when teams are not enabled or the guarantee holds; throws otherwise unless
// KOLM_ALLOW_NONTXN_TEAMS=true explicitly accepts the risk.
export function assertTeamsTransactionality({ teamsEnabled = true } = {}) {
  const info = { ok: true, driver: STORE_DRIVER, transactional: isTransactional() };
  if (!teamsEnabled) return info;
  const prodLike = isProductionRuntime();
  if (prodLike && !isTransactional() && process.env.KOLM_ALLOW_NONTXN_TEAMS !== 'true') {
    throw new Error(
      `[store] FATAL: Teams seat billing requires a transactional core store, but KOLM_STORE_DRIVER resolved to "${STORE_DRIVER}". `
      + 'withTransaction (BEGIN IMMEDIATE) is sqlite-only; on json the seat TOCTOU guard is not a real serialization barrier. '
      + 'Set KOLM_STORE_DRIVER=sqlite, or set KOLM_ALLOW_NONTXN_TEAMS=true to explicitly accept non-serialized seat accounting.',
    );
  }
  return info;
}

// True when the store fell back to ephemeral /tmp (non-production only; in
// production this condition throws at boot). Surfaced in /health so an operator
// sees the degraded state.
export function storeEphemeral() {
  return STORE_EPHEMERAL;
}

// =============================================================================
// W808-2 - staged_captures table (capture quarantine).
//
// New rows from the proxy capture path land HERE first (not in the canonical
// `observations` table). Each row carries a `quarantine_until` deadline
// (default now + 24h) plus optional `anomaly_flagged` + `manual_block_reason`
// fields. promoteStagedCapture(id) lifts a row into `observations` once it
// has cleared:
//   (a) no anomaly flag (or operator override), AND
//   (b) no manual block, AND
//   (c) quarantine_until <= now (or operator override via auto_allow_since).
//
// This module DOES NOT touch the existing `observations` table schema. It
// only adds parallel staging + a one-way promotion verb. If the staged
// table is empty the proxy can fall back to direct-write into observations
// (when KOLM_W808_STAGING=0) - that env-gate lives in src/proxy.js, not
// here.
//
// All getters/setters are tenant-scoped via the canonical `tenant_id` field
// (W411). The driver itself does not enforce the fence - callers must.
// =============================================================================

// W808-2 default quarantine window. Exported so the proxy + tests can
// reference the same constant.
export const W808_DEFAULT_QUARANTINE_MS = 24 * 60 * 60 * 1000; // 24h
export const W808_STAGED_TABLE = 'staged_captures';

// Insert a new staged capture row. Mints a synthetic id if absent.
// Always stamps quarantine_until (now+24h by default). Returns the row.
//
// Callers MUST pass row.tenant_id - we error out otherwise to keep the
// W411 fence loud (silent default-tenant rows poison the whole baseline).
export function insertStagedCapture(row, opts = {}) {
  if (!row || typeof row !== 'object') {
    throw new Error('insertStagedCapture: row must be an object');
  }
  if (!row.tenant_id && !row.tenant) {
    throw new Error('insertStagedCapture: row.tenant_id is required (W411 tenant fence)');
  }
  const quarantineMs = Number.isFinite(opts.quarantine_ms) && opts.quarantine_ms > 0
    ? opts.quarantine_ms : W808_DEFAULT_QUARANTINE_MS;
  const staged = {
    ...row,
    staged_capture_id: row.staged_capture_id || id('stg'),
    staged_at: row.staged_at || new Date().toISOString(),
    quarantine_until: row.quarantine_until || new Date(Date.now() + quarantineMs).toISOString(),
    quarantine_state: row.quarantine_state || 'pending',
    anomaly_flagged: row.anomaly_flagged === true,
    anomaly_reasons: Array.isArray(row.anomaly_reasons) ? row.anomaly_reasons : [],
    manual_block_reason: row.manual_block_reason || null,
    manual_review_at: row.manual_review_at || null,
    manual_review_by: row.manual_review_by || null,
    w808_version: 'w808-v1',
  };
  insert(W808_STAGED_TABLE, staged);
  return staged;
}

// List staged captures pending review. Filters: tenant_id (required),
// namespace (optional), includeBlocked (default false), limit (default 500).
// Returns newest-first by staged_at.
export function listStagedCaptures({ tenant_id, namespace = null, includeBlocked = false, limit = 500 } = {}) {
  if (!tenant_id) return [];
  const rows = findByField(W808_STAGED_TABLE, 'tenant_id', tenant_id);
  const out = [];
  for (const r of rows) {
    // Inner-loop tenant fence - never trust the index alone (W411).
    if (String(r.tenant_id) !== String(tenant_id)) continue;
    if (namespace && String(r.namespace || r.corpus_namespace || 'default') !== String(namespace)) continue;
    if (!includeBlocked && r.quarantine_state === 'blocked') continue;
    if (r.quarantine_state === 'promoted') continue; // already in observations
    out.push(r);
  }
  out.sort((a, b) => String(b.staged_at || '').localeCompare(String(a.staged_at || '')));
  return out.slice(0, Math.max(1, limit));
}

// Get one staged capture by staged_capture_id. Returns null if missing OR
// if the caller's tenant_id does not match the row (W411).
export function getStagedCapture(staged_capture_id, { tenant_id } = {}) {
  if (!staged_capture_id) return null;
  const rows = findByField(W808_STAGED_TABLE, 'staged_capture_id', staged_capture_id);
  for (const r of rows) {
    if (tenant_id && String(r.tenant_id) !== String(tenant_id)) continue;
    return r;
  }
  return null;
}

// Mark a staged row as anomaly_flagged (called from the proxy after the
// anomaly detector returns ok:true + anomaly_flagged:true). Returns the
// number of rows patched (0 or 1).
export function markStagedAnomaly(staged_capture_id, { tenant_id, reasons = [], flagged_axes = [] } = {}) {
  if (!staged_capture_id) return 0;
  return update(W808_STAGED_TABLE,
    (r) => r.staged_capture_id === staged_capture_id
      && (!tenant_id || String(r.tenant_id) === String(tenant_id)),
    {
      anomaly_flagged: true,
      anomaly_reasons: Array.isArray(reasons) ? reasons : [],
      anomaly_flagged_axes: Array.isArray(flagged_axes) ? flagged_axes : [],
      anomaly_flagged_at: new Date().toISOString(),
    });
}

// Manual block - operator chose to refuse this capture forever. Returns 0/1.
export function blockStagedCapture(staged_capture_id, { tenant_id, reason, reviewer = null } = {}) {
  if (!staged_capture_id) return 0;
  if (!reason || typeof reason !== 'string') {
    throw new Error('blockStagedCapture: reason is required (audit trail)');
  }
  return update(W808_STAGED_TABLE,
    (r) => r.staged_capture_id === staged_capture_id
      && (!tenant_id || String(r.tenant_id) === String(tenant_id)),
    {
      quarantine_state: 'blocked',
      manual_block_reason: reason,
      manual_review_at: new Date().toISOString(),
      manual_review_by: reviewer || null,
    });
}

// Manual allow - operator chose to override anomaly flag / quarantine timer
// and promote NOW. Returns the promoted row (or null if not found / blocked).
// The actual insert into `observations` is delegated to a callback so this
// module stays decoupled from src/capture-store.js.
export function promoteStagedCapture(staged_capture_id, { tenant_id, reviewer = null, force = false, insertObservation } = {}) {
  if (!staged_capture_id) return null;
  const row = getStagedCapture(staged_capture_id, { tenant_id });
  if (!row) return null;
  if (row.quarantine_state === 'blocked' && !force) return null;
  if (row.quarantine_state === 'promoted') return row;
  if (!force) {
    // No-anomaly-flag AND no-block (W808-2 contract).
    if (row.anomaly_flagged === true) return null;
    if (row.manual_block_reason) return null;
    // Quarantine deadline must have passed (operator can force).
    const deadline = Date.parse(row.quarantine_until || '');
    if (Number.isFinite(deadline) && Date.now() < deadline) return null;
  }
  // Promote - delegate the insert; mark staged row as promoted on success.
  if (typeof insertObservation === 'function') {
    try { insertObservation(row); } catch (e) {
      // Re-throw with context so the caller's audit trail records the failure.
      const err = new Error(`promoteStagedCapture: observation insert failed: ${e.message || e}`);
      err.cause = e;
      throw err;
    }
  }
  update(W808_STAGED_TABLE,
    (r) => r.staged_capture_id === staged_capture_id
      && (!tenant_id || String(r.tenant_id) === String(tenant_id)),
    {
      quarantine_state: 'promoted',
      manual_review_at: new Date().toISOString(),
      manual_review_by: reviewer || row.manual_review_by || null,
    });
  return { ...row, quarantine_state: 'promoted' };
}

// Auto-allow sweep - promote every staged row whose quarantine_until has
// elapsed AND that carries no anomaly flag AND no block. Returns
// { promoted, skipped, blocked, anomalous }.
export function autoAllowSinceQuarantine({ tenant_id, since_ms = W808_DEFAULT_QUARANTINE_MS, insertObservation } = {}) {
  if (!tenant_id) return { promoted: 0, skipped: 0, blocked: 0, anomalous: 0 };
  const now = Date.now();
  const cutoff = now - Math.max(0, since_ms);
  const rows = listStagedCaptures({ tenant_id, includeBlocked: true, limit: 5000 });
  let promoted = 0, skipped = 0, blocked = 0, anomalous = 0;
  for (const r of rows) {
    if (r.quarantine_state === 'blocked') { blocked += 1; continue; }
    if (r.quarantine_state === 'promoted') continue;
    if (r.anomaly_flagged === true) { anomalous += 1; continue; }
    if (r.manual_block_reason) { blocked += 1; continue; }
    const staged = Date.parse(r.staged_at || '');
    if (!Number.isFinite(staged) || staged > cutoff) { skipped += 1; continue; }
    const promoted_row = promoteStagedCapture(r.staged_capture_id, {
      tenant_id,
      reviewer: 'auto-allow',
      force: true,                 // we've already validated the gates above
      insertObservation,
    });
    if (promoted_row) promoted += 1;
    else skipped += 1;
  }
  return { promoted, skipped, blocked, anomalous };
}

// W-5 - the single capture-routing decision the proxy makes, and the call site
// that was missing (insertStagedCapture had ZERO callers, so quarantine was dead
// for proxy traffic). Default (staging disabled) is the historical behavior:
// write straight to `observations` via the caller's insertObservation. When
// KOLM_W808_STAGING is enabled the row is quarantined in staged_captures instead,
// carrying any anomaly/copyright/manual flags, and only reaches observations via
// promoteStagedCapture / the auto-allow sweep. Gating on KOLM_W808_STAGING keeps
// enabling quarantine a deliberate operator choice, never a silent default.
export function stageOrPassthrough({
  row,
  stagingEnabled = false,
  anomalyFlagged = false,
  anomalyReasons = [],
  manualBlockReason = null,
  insertObservation,
} = {}) {
  if (!row || typeof row !== 'object') {
    throw new Error('stageOrPassthrough: row must be an object');
  }
  if (stagingEnabled) {
    const staged = insertStagedCapture({
      ...row,
      anomaly_flagged: anomalyFlagged === true,
      anomaly_reasons: Array.isArray(anomalyReasons) ? anomalyReasons : [],
      manual_block_reason: manualBlockReason || null,
    });
    return { staged: true, row: staged };
  }
  if (typeof insertObservation === 'function') insertObservation(row);
  return { staged: false, row };
}

// Reset hook for tests - empties the staged_captures table.
export function _resetStagedCapturesForTests() {
  remove(W808_STAGED_TABLE, () => true);
}
