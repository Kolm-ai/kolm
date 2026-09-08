// kolm code — ladder.db state store (spec 5.1). Owner C.
//
// node:sqlite, WAL, synchronous API. One database per machine holds every run:
// runs, candidates, per-stage verdicts (versioned, never deleted), judge
// verdicts, the per-SHA context cache, the event log the viewer replays, and
// the cost ledger. Re-running a bumped stage version ADDS rows; nothing is ever
// overwritten in place except an explicit re-put of the same (stage, version).

import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { createRequire } from 'node:module';
import { LadderError } from './errors.js';
import { funnel as funnelMath } from './verdict.js';

const require = createRequire(import.meta.url);

export const SCHEMA_VERSION = 1;

/**
 * True when this Node can `require('node:sqlite')` (Node >= 22.5).
 * Every ladder test file that needs the store skips itself when this is false,
 * so CI on Node 20 stays green.
 * @returns {boolean}
 */
export function sqliteAvailable() {
  try {
    const mod = require('node:sqlite');
    return typeof mod?.DatabaseSync === 'function';
  } catch {
    return false;
  }
}

const SCHEMA = `
CREATE TABLE IF NOT EXISTS runs(
  id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  status TEXT,
  task_json TEXT,
  repo_path TEXT,
  base_sha TEXT,
  driver TEXT,
  mode TEXT,
  config_json TEXT,
  winner_candidate TEXT,
  total_cost_usd REAL DEFAULT 0,
  receipt_json TEXT
);
CREATE TABLE IF NOT EXISTS candidates(
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  candidate_id TEXT NOT NULL,
  idx INTEGER DEFAULT 0,
  attempt INTEGER DEFAULT 0,
  generator_ref TEXT,
  worktree_json TEXT,
  status TEXT,
  diff_sha256 TEXT,
  diff_text TEXT,
  summary TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS results(
  run_id TEXT NOT NULL,
  candidate_id TEXT NOT NULL,
  stage TEXT NOT NULL,
  version INTEGER NOT NULL,
  verdict_json TEXT,
  created_at TEXT NOT NULL,
  PRIMARY KEY(run_id, candidate_id, stage, version)
);
CREATE TABLE IF NOT EXISTS judgments(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  candidate_id TEXT,
  kind TEXT,
  verdict_json TEXT,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS context_cache(
  repo_key TEXT NOT NULL,
  sha TEXT NOT NULL,
  kind TEXT NOT NULL,
  payload_json TEXT,
  created_at TEXT NOT NULL,
  PRIMARY KEY(repo_key, sha, kind)
);
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT,
  ts TEXT NOT NULL,
  type TEXT,
  payload_json TEXT
);
CREATE TABLE IF NOT EXISTS cost_ledger(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT,
  candidate_id TEXT,
  stage TEXT,
  provider TEXT,
  model TEXT,
  input_tokens INTEGER DEFAULT 0,
  output_tokens INTEGER DEFAULT 0,
  cache_read INTEGER DEFAULT 0,
  cost_usd REAL DEFAULT 0,
  ts TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS schema_meta(
  key TEXT PRIMARY KEY,
  value TEXT
);
CREATE INDEX IF NOT EXISTS idx_results_run ON results(run_id);
CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id, id);
CREATE INDEX IF NOT EXISTS idx_cost_run ON cost_ledger(run_id);
CREATE INDEX IF NOT EXISTS idx_candidates_run ON candidates(run_id);
CREATE INDEX IF NOT EXISTS idx_judgments_run ON judgments(run_id, candidate_id);
`;

const UPDATE_RUN_KEYS = new Map([
  ['status', 'status'],
  ['base_sha', 'base_sha'],
  ['baseSha', 'base_sha'],
  ['winner_candidate', 'winner_candidate'],
  ['winnerCandidate', 'winner_candidate'],
  ['total_cost_usd', 'total_cost_usd'],
  ['totalCostUsd', 'total_cost_usd'],
  ['receipt_json', 'receipt_json'],
  ['receipt', 'receipt_json'],
]);

function nowIso() {
  return new Date().toISOString();
}

function toJson(value) {
  if (value === undefined || value === null) return null;
  if (typeof value === 'string') return value;
  return JSON.stringify(value);
}

function fromJson(text) {
  if (text === undefined || text === null || text === '') return null;
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

function num(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

/** `r_<yyyymmdd>_<8 hex>` */
export function newRunId(date = new Date()) {
  const y = date.getUTCFullYear();
  const m = String(date.getUTCMonth() + 1).padStart(2, '0');
  const d = String(date.getUTCDate()).padStart(2, '0');
  return `r_${y}${m}${d}_${crypto.randomBytes(4).toString('hex')}`;
}

/** `c<idx>a<attempt>` */
export function candidateIdFor(idx, attempt = 0) {
  return `c${idx}a${attempt}`;
}

function rowKey(runId, candidateId) {
  return `${runId}/${candidateId}`;
}

function shapeRun(row) {
  if (!row) return null;
  return {
    id: row.id,
    created_at: row.created_at,
    updated_at: row.updated_at,
    status: row.status,
    task: fromJson(row.task_json),
    repo_path: row.repo_path,
    base_sha: row.base_sha,
    driver: row.driver,
    mode: row.mode,
    config: fromJson(row.config_json),
    winner_candidate: row.winner_candidate,
    total_cost_usd: num(row.total_cost_usd),
    receipt: fromJson(row.receipt_json),
  };
}

function shapeCandidate(row) {
  if (!row) return null;
  return {
    id: row.id,
    run_id: row.run_id,
    candidate_id: row.candidate_id,
    idx: num(row.idx),
    attempt: num(row.attempt),
    generator_ref: row.generator_ref,
    worktree: fromJson(row.worktree_json),
    status: row.status,
    diff_sha256: row.diff_sha256,
    diff_text: row.diff_text,
    summary: row.summary,
    created_at: row.created_at,
    updated_at: row.updated_at,
  };
}

/**
 * Open (creating if needed) the ladder database.
 * @param {string} dbPath
 * @returns {object} store
 */
export function openStore(dbPath) {
  let DatabaseSync;
  try {
    ({ DatabaseSync } = require('node:sqlite'));
  } catch {
    throw new LadderError(
      'sqlite-unavailable',
      `node:sqlite is unavailable on Node ${process.version}; kolm code needs Node >= 22.5`,
      { node: process.version },
    );
  }
  if (typeof DatabaseSync !== 'function') {
    throw new LadderError(
      'sqlite-unavailable',
      `node:sqlite is unavailable on Node ${process.version}; kolm code needs Node >= 22.5`,
      { node: process.version },
    );
  }
  const resolved = path.resolve(dbPath);
  fs.mkdirSync(path.dirname(resolved), { recursive: true });
  const db = new DatabaseSync(resolved);
  db.exec('PRAGMA journal_mode=WAL');
  db.exec('PRAGMA busy_timeout=5000');
  db.exec('PRAGMA foreign_keys=ON');
  db.exec(SCHEMA);
  db.prepare('INSERT OR REPLACE INTO schema_meta(key, value) VALUES(?, ?)').run('schema_version', String(SCHEMA_VERSION));

  const stmt = (sql) => db.prepare(sql);

  const insertRun = stmt(`INSERT INTO runs(id, created_at, updated_at, status, task_json, repo_path,
    base_sha, driver, mode, config_json, winner_candidate, total_cost_usd, receipt_json)
    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)`);
  const selectRun = stmt('SELECT * FROM runs WHERE id = ?');
  const selectCandidate = stmt('SELECT * FROM candidates WHERE id = ?');
  const insertCandidate = stmt(`INSERT INTO candidates(id, run_id, candidate_id, idx, attempt, generator_ref,
    worktree_json, status, diff_sha256, diff_text, summary, created_at, updated_at)
    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)`);
  const listCandidatesStmt = stmt('SELECT * FROM candidates WHERE run_id = ? ORDER BY idx ASC, attempt ASC, candidate_id ASC');
  const putResultStmt = stmt(`INSERT OR REPLACE INTO results(run_id, candidate_id, stage, version, verdict_json, created_at)
    VALUES(?,?,?,?,?,?)`);
  const getResultStmt = stmt('SELECT * FROM results WHERE run_id = ? AND candidate_id = ? AND stage = ? AND version = ?');
  const latestResultsStmt = stmt('SELECT * FROM results WHERE run_id = ? AND candidate_id = ? ORDER BY stage ASC, version ASC');
  const resultsForStmt = stmt('SELECT * FROM results WHERE run_id = ? ORDER BY candidate_id ASC, stage ASC, version ASC');
  const insertJudgment = stmt('INSERT INTO judgments(run_id, candidate_id, kind, verdict_json, created_at) VALUES(?,?,?,?,?)');
  const cacheGetStmt = stmt('SELECT payload_json FROM context_cache WHERE repo_key = ? AND sha = ? AND kind = ?');
  const cachePutStmt = stmt('INSERT OR REPLACE INTO context_cache(repo_key, sha, kind, payload_json, created_at) VALUES(?,?,?,?,?)');
  const insertEvent = stmt('INSERT INTO events(run_id, ts, type, payload_json) VALUES(?,?,?,?)');
  const eventsSinceStmt = stmt('SELECT * FROM events WHERE run_id = ? AND id > ? ORDER BY id ASC LIMIT ?');
  const insertCost = stmt(`INSERT INTO cost_ledger(run_id, candidate_id, stage, provider, model,
    input_tokens, output_tokens, cache_read, cost_usd, ts) VALUES(?,?,?,?,?,?,?,?,?,?)`);
  const costRowsStmt = stmt('SELECT * FROM cost_ledger WHERE run_id = ? ORDER BY id ASC');

  let closed = false;
  const assertOpen = () => {
    if (closed) throw new LadderError('store-closed', 'the ladder store is closed');
  };

  const store = {
    /** absolute path of the sqlite file */
    path: resolved,
    /** escape hatch for owner D's receipt/eval code; do not use for writes */
    db,

    // ---------------------------------------------------------------- runs
    createRun({ id, task, repoPath, baseSha, driver, mode, config, status = 'running' } = {}) {
      assertOpen();
      const runId = id || newRunId();
      const ts = nowIso();
      insertRun.run(
        runId,
        ts,
        ts,
        status,
        toJson(task),
        repoPath ?? null,
        baseSha ?? null,
        driver ?? null,
        mode ?? null,
        toJson(config),
        null,
        0,
        null,
      );
      return shapeRun(selectRun.get(runId));
    },

    updateRun(id, patch = {}) {
      assertOpen();
      const sets = [];
      const args = [];
      for (const [key, value] of Object.entries(patch)) {
        const column = UPDATE_RUN_KEYS.get(key);
        if (!column) continue;
        if (sets.includes(`${column} = ?`)) continue;
        sets.push(`${column} = ?`);
        if (column === 'receipt_json') args.push(toJson(value));
        else if (column === 'total_cost_usd') args.push(num(value));
        else args.push(value ?? null);
      }
      sets.push('updated_at = ?');
      args.push(nowIso());
      args.push(id);
      db.prepare(`UPDATE runs SET ${sets.join(', ')} WHERE id = ?`).run(...args);
      return shapeRun(selectRun.get(id));
    },

    getRun(id) {
      assertOpen();
      return shapeRun(selectRun.get(id));
    },

    listRuns({ limit = 50, status } = {}) {
      assertOpen();
      const args = [];
      let sql = 'SELECT * FROM runs';
      if (status) {
        sql += ' WHERE status = ?';
        args.push(status);
      }
      sql += ' ORDER BY created_at DESC, rowid DESC LIMIT ?';
      args.push(Math.max(1, Math.trunc(num(limit, 50))));
      return db.prepare(sql).all(...args).map(shapeRun);
    },

    // ---------------------------------------------------------- candidates
    upsertCandidate({ runId, candidateId, idx, attempt, generatorRef, worktree, status, diffText, summary } = {}) {
      assertOpen();
      if (!runId || !candidateId) {
        throw new LadderError('bad-arg', 'upsertCandidate needs runId and candidateId');
      }
      const key = rowKey(runId, candidateId);
      const ts = nowIso();
      const existing = selectCandidate.get(key);
      const diffSha = diffText === undefined || diffText === null
        ? undefined
        : crypto.createHash('sha256').update(String(diffText)).digest('hex');
      if (!existing) {
        insertCandidate.run(
          key,
          runId,
          candidateId,
          Math.trunc(num(idx, 0)),
          Math.trunc(num(attempt, 0)),
          generatorRef ?? null,
          toJson(worktree),
          status ?? null,
          diffSha ?? null,
          diffText ?? null,
          summary ?? null,
          ts,
          ts,
        );
        return shapeCandidate(selectCandidate.get(key));
      }
      const sets = [];
      const args = [];
      const put = (column, value) => {
        if (value === undefined) return;
        sets.push(`${column} = ?`);
        args.push(value);
      };
      put('idx', idx === undefined ? undefined : Math.trunc(num(idx, 0)));
      put('attempt', attempt === undefined ? undefined : Math.trunc(num(attempt, 0)));
      put('generator_ref', generatorRef);
      put('worktree_json', worktree === undefined ? undefined : toJson(worktree));
      put('status', status);
      put('diff_sha256', diffSha);
      put('diff_text', diffText);
      put('summary', summary);
      sets.push('updated_at = ?');
      args.push(ts);
      args.push(key);
      db.prepare(`UPDATE candidates SET ${sets.join(', ')} WHERE id = ?`).run(...args);
      return shapeCandidate(selectCandidate.get(key));
    },

    getCandidate(runId, candidateId) {
      assertOpen();
      return shapeCandidate(selectCandidate.get(rowKey(runId, candidateId)));
    },

    listCandidates(runId) {
      assertOpen();
      return listCandidatesStmt.all(runId).map(shapeCandidate);
    },

    // ------------------------------------------------------------- results
    putResult({ runId, candidateId, stage, version, verdict } = {}) {
      assertOpen();
      if (!runId || !candidateId || !stage) {
        throw new LadderError('bad-arg', 'putResult needs runId, candidateId and stage');
      }
      const ts = nowIso();
      putResultStmt.run(runId, candidateId, stage, Math.trunc(num(version, 1)), toJson(verdict), ts);
      return { runId, candidateId, stage, version: Math.trunc(num(version, 1)), created_at: ts };
    },

    getResult(runId, candidateId, stage, version) {
      assertOpen();
      const row = getResultStmt.get(runId, candidateId, stage, Math.trunc(num(version, 1)));
      return row ? fromJson(row.verdict_json) : null;
    },

    latestResults(runId, candidateId) {
      assertOpen();
      const out = {};
      for (const row of latestResultsStmt.all(runId, candidateId)) {
        const version = num(row.version);
        const prev = out[row.stage];
        if (!prev || version >= prev.version) {
          out[row.stage] = { version, verdict: fromJson(row.verdict_json), created_at: row.created_at };
        }
      }
      return out;
    },

    resultsFor(runId) {
      assertOpen();
      return resultsForStmt.all(runId).map((row) => ({
        run_id: row.run_id,
        candidate_id: row.candidate_id,
        stage: row.stage,
        version: num(row.version),
        verdict: fromJson(row.verdict_json),
        created_at: row.created_at,
      }));
    },

    // ----------------------------------------------------------- judgments
    putJudgment({ runId, candidateId, kind, verdict } = {}) {
      assertOpen();
      const ts = nowIso();
      const info = insertJudgment.run(runId ?? null, candidateId ?? null, kind ?? null, toJson(verdict), ts);
      return Number(info.lastInsertRowid);
    },

    listJudgments(runId, candidateId) {
      assertOpen();
      const args = [runId];
      let sql = 'SELECT * FROM judgments WHERE run_id = ?';
      if (candidateId !== undefined && candidateId !== null) {
        sql += ' AND candidate_id = ?';
        args.push(candidateId);
      }
      sql += ' ORDER BY id ASC';
      return db.prepare(sql).all(...args).map((row) => ({
        id: num(row.id),
        run_id: row.run_id,
        candidate_id: row.candidate_id,
        kind: row.kind,
        verdict: fromJson(row.verdict_json),
        created_at: row.created_at,
      }));
    },

    // ------------------------------------------------------- context cache
    cacheGet(repoKey, sha, kind) {
      assertOpen();
      const row = cacheGetStmt.get(String(repoKey), String(sha), String(kind));
      return row ? fromJson(row.payload_json) : null;
    },

    cachePut(repoKey, sha, kind, payload) {
      assertOpen();
      cachePutStmt.run(String(repoKey), String(sha), String(kind), toJson(payload), nowIso());
      return payload;
    },

    // -------------------------------------------------------------- events
    emit(runId, type, payload) {
      assertOpen();
      const info = insertEvent.run(runId ?? null, nowIso(), String(type), toJson(payload));
      return Number(info.lastInsertRowid);
    },

    eventsSince(runId, afterId = 0, limit = 500) {
      assertOpen();
      return eventsSinceStmt
        .all(runId, Math.trunc(num(afterId, 0)), Math.max(1, Math.trunc(num(limit, 500))))
        .map((row) => ({
          id: num(row.id),
          run_id: row.run_id,
          ts: row.ts,
          type: row.type,
          payload: fromJson(row.payload_json),
        }));
    },

    // ----------------------------------------------------------------- cost
    addCost({ runId, candidateId, stage, provider, model, inputTokens, outputTokens, cacheRead, costUsd } = {}) {
      assertOpen();
      const info = insertCost.run(
        runId ?? null,
        candidateId ?? null,
        stage ?? null,
        provider ?? null,
        model ?? null,
        Math.trunc(num(inputTokens, 0)),
        Math.trunc(num(outputTokens, 0)),
        Math.trunc(num(cacheRead, 0)),
        num(costUsd, 0),
        nowIso(),
      );
      return Number(info.lastInsertRowid);
    },

    costSummary(runId) {
      assertOpen();
      const byModel = {};
      const byStage = {};
      let total = 0;
      for (const row of costRowsStmt.all(runId)) {
        const cost = num(row.cost_usd);
        total += cost;
        const modelKey = `${row.provider || 'unknown'}:${row.model || 'unknown'}`;
        const stageKey = row.stage || 'unknown';
        for (const [map, key] of [[byModel, modelKey], [byStage, stageKey]]) {
          const bucket = (map[key] ||= { input_tokens: 0, output_tokens: 0, cache_read: 0, cost_usd: 0, calls: 0 });
          bucket.input_tokens += num(row.input_tokens);
          bucket.output_tokens += num(row.output_tokens);
          bucket.cache_read += num(row.cache_read);
          bucket.cost_usd += cost;
          bucket.calls += 1;
        }
      }
      return { total_usd: total, by_model: byModel, by_stage: byStage };
    },

    // --------------------------------------------------------------- funnel
    funnel({ runIds, since } = {}) {
      assertOpen();
      const where = [];
      const args = [];
      if (Array.isArray(runIds) && runIds.length > 0) {
        where.push(`run_id IN (${runIds.map(() => '?').join(',')})`);
        args.push(...runIds);
      }
      if (since) {
        where.push('created_at >= ?');
        args.push(String(since));
      }
      const clause = where.length > 0 ? ` WHERE ${where.join(' AND ')}` : '';
      const resultRows = db
        .prepare(`SELECT run_id, candidate_id, stage, version, verdict_json FROM results${clause} ORDER BY version ASC`)
        .all(...args);
      // Latest version per (run, candidate, stage).
      const latest = new Map();
      for (const row of resultRows) {
        const key = `${row.run_id}\u0000${row.candidate_id}\u0000${row.stage}`;
        const prev = latest.get(key);
        if (!prev || num(row.version) >= prev.version) {
          latest.set(key, { ...row, version: num(row.version) });
        }
      }
      const judgedPairs = new Set();
      const judgeRows = db.prepare('SELECT run_id, candidate_id FROM judgments').all();
      for (const row of judgeRows) judgedPairs.add(`${row.run_id}\u0000${row.candidate_id}`);

      const rows = [];
      for (const row of latest.values()) {
        const verdict = fromJson(row.verdict_json) || {};
        rows.push({
          runId: row.run_id,
          candidateId: row.candidate_id,
          stage: row.stage,
          decision: verdict.decision || 'skip',
          judged: judgedPairs.has(`${row.run_id}\u0000${row.candidate_id}`),
        });
      }
      return funnelMath(rows);
    },

    close() {
      if (closed) return;
      closed = true;
      try {
        db.close();
      } catch {
        // already closed by the caller / process teardown
      }
    },
  };
  return store;
}

export default { openStore, sqliteAvailable, newRunId, candidateIdFor, SCHEMA_VERSION };
