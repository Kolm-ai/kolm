// kolm code — viewer HTTP server (owner E, spec section 7).
//
// `node:http` only: no express, no dependency, no build step. The server is a
// read-only window onto one ladder store. It never writes a row, never spawns a
// provider call and never reaches the network; every response is derived from
// the store handle it was given.
//
// Binding is 127.0.0.1 by default and refuses a wildcard bind unless the
// operator opts in with KOLM_CODE_VIEWER_PUBLIC=1, because a run's event log
// carries task text and diffs.

import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { LadderError } from '../errors.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));

/** The single-file viewer shipped next to this module. */
export const DEFAULT_INDEX_HTML = path.join(HERE, 'index.html');

/** Content-Security-Policy for the HTML document (spec section 7). */
export const VIEWER_CSP =
  "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src 'self' data:";

const WILDCARD_HOSTS = new Set(['0.0.0.0', '::', '[::]', '*']);

function jsonHeaders(extra = {}) {
  return {
    'Content-Type': 'application/json; charset=utf-8',
    'Cache-Control': 'no-store',
    ...extra,
  };
}

function sendJson(res, status, body) {
  let text;
  try {
    text = JSON.stringify(body);
  } catch {
    status = 500;
    text = JSON.stringify({ error: 'unserializable', message: 'response could not be encoded' });
  }
  res.writeHead(status, jsonHeaders({ 'Content-Length': Buffer.byteLength(text) }));
  res.end(text);
}

function notFound(res, what) {
  sendJson(res, 404, { error: 'not-found', message: `no such resource: ${what}` });
}

function intParam(value, fallback) {
  // Number(null) is 0, so an absent query parameter must be caught first.
  if (value === null || value === undefined || value === '') return fallback;
  const n = Number(value);
  return Number.isFinite(n) ? Math.trunc(n) : fallback;
}

/**
 * Everything the viewer needs about one run in a single payload.
 * @param {object} store
 * @param {string} runId
 */
function runDetail(store, runId) {
  const run = store.getRun(runId);
  if (!run) return null;
  const candidates = store.listCandidates(runId).map((candidate) => ({
    ...candidate,
    // The raw diff can be megabytes; the viewer only needs its size.
    diff_text: undefined,
    diff_bytes: candidate.diff_text ? Buffer.byteLength(candidate.diff_text) : 0,
    results: store.latestResults(runId, candidate.candidate_id),
  }));
  return {
    run,
    candidates,
    judgments: store.listJudgments(runId),
    cost: store.costSummary(runId),
    funnel: store.funnel({ runIds: [runId] }),
    events: store.eventsSince(runId, 0, 1).length,
  };
}

/**
 * Create (but do not start) the viewer server.
 *
 * @param {object} opts
 * @param {object} opts.store            an open store from `openStore(dbPath)`
 * @param {object} [opts.config]         ladder config (only `dbPath` is read, for /api/doctor)
 * @param {string} [opts.host]           default 127.0.0.1
 * @param {number} [opts.port]           default 0 (ephemeral)
 * @param {string} [opts.indexHtmlPath]  override the served HTML
 * @param {Function} [opts.verifyReceipt] owner D's `verifyReceipt`
 * @param {Function} [opts.evalRuns]     owner D's `evalRuns`
 * @param {Function} [opts.arming]       returns an arming report for /api/doctor
 * @param {number} [opts.pollMs]         SSE poll interval (default 500)
 * @param {number} [opts.heartbeatMs]    SSE comment heartbeat (default 15000)
 * @param {Record<string,string>} [opts.env] environment used for the public-bind opt-in
 * @returns {{server: import('node:http').Server, host: string, listen: Function, close: Function, url: string|null}}
 */
export function createViewerServer({
  store,
  config = {},
  host = '127.0.0.1',
  port = 0,
  indexHtmlPath = DEFAULT_INDEX_HTML,
  verifyReceipt = null,
  evalRuns = null,
  arming = null,
  pollMs = 500,
  heartbeatMs = 15000,
  env = process.env,
} = {}) {
  if (!store || typeof store.listRuns !== 'function') {
    throw new LadderError('bad-arg', 'createViewerServer needs an open ladder store', {});
  }
  if (WILDCARD_HOSTS.has(String(host)) && String(env?.KOLM_CODE_VIEWER_PUBLIC ?? '') !== '1') {
    throw new LadderError(
      'bad-arg',
      `refusing to bind the viewer to ${host}: run logs carry task text and diffs. ` +
        'Set KOLM_CODE_VIEWER_PUBLIC=1 to override.',
      { host },
    );
  }

  /** @type {Set<import('node:net').Socket>} */
  const sockets = new Set();
  /** @type {Set<{res: object, timers: Array}>} */
  const streams = new Set();
  let listeningPort = null;

  function closeStream(entry) {
    if (!streams.has(entry)) return;
    streams.delete(entry);
    for (const timer of entry.timers) clearInterval(timer);
    try {
      entry.res.end();
    } catch {
      // the client is already gone; nothing else to do
    }
    try {
      entry.res.destroy();
    } catch {
      // ditto
    }
  }

  function serveIndex(res) {
    let html;
    try {
      html = fs.readFileSync(indexHtmlPath);
    } catch (err) {
      sendJson(res, 500, {
        error: 'viewer-html-missing',
        message: `cannot read ${indexHtmlPath}: ${err.code || err.message}`,
      });
      return;
    }
    res.writeHead(200, {
      'Content-Type': 'text/html; charset=utf-8',
      'Cache-Control': 'no-store',
      'Content-Security-Policy': VIEWER_CSP,
      'X-Content-Type-Options': 'nosniff',
      'Referrer-Policy': 'no-referrer',
      'Content-Length': html.length,
    });
    res.end(html);
  }

  function openStream(req, res, runId) {
    const lastEventId = intParam(req.headers['last-event-id'], 0);
    let cursor = Math.max(0, lastEventId);
    res.writeHead(200, {
      'Content-Type': 'text/event-stream; charset=utf-8',
      'Cache-Control': 'no-store',
      Connection: 'keep-alive',
      'X-Accel-Buffering': 'no',
    });
    res.write(': kolm code event stream\n\n');

    const entry = { res, timers: [] };
    streams.add(entry);

    const pump = () => {
      if (!streams.has(entry)) return;
      let batch;
      try {
        batch = store.eventsSince(runId, cursor, 200);
      } catch {
        closeStream(entry);
        return;
      }
      for (const event of batch) {
        cursor = Math.max(cursor, event.id);
        let payload;
        try {
          payload = JSON.stringify(event);
        } catch {
          payload = JSON.stringify({ id: event.id, type: event.type, payload: null });
        }
        res.write(`id: ${event.id}\nevent: ${event.type || 'message'}\ndata: ${payload}\n\n`);
      }
    };

    pump();
    entry.timers.push(setInterval(pump, Math.max(50, pollMs)));
    entry.timers.push(
      setInterval(() => {
        if (streams.has(entry)) res.write(': heartbeat\n\n');
      }, Math.max(1000, heartbeatMs)),
    );
    for (const timer of entry.timers) if (typeof timer.unref === 'function') timer.unref();
    req.on('close', () => closeStream(entry));
    res.on('close', () => closeStream(entry));
  }

  const server = http.createServer((req, res) => {
    let url;
    try {
      url = new URL(req.url || '/', 'http://viewer.local');
    } catch {
      sendJson(res, 400, { error: 'bad-request', message: 'unparseable request target' });
      return;
    }
    const pathname = url.pathname.replace(/\/+$/, '') || '/';

    if (req.method !== 'GET' && req.method !== 'HEAD') {
      sendJson(res, 405, { error: 'method-not-allowed', message: `${req.method} is not served` });
      return;
    }

    try {
      if (pathname === '/' || pathname === '/index.html') {
        serveIndex(res);
        return;
      }

      if (pathname === '/api/runs') {
        const limit = intParam(url.searchParams.get('limit'), 50);
        sendJson(res, 200, { runs: store.listRuns({ limit: Math.max(1, limit) }) });
        return;
      }

      if (pathname === '/api/funnel') {
        const since = url.searchParams.get('since') || undefined;
        sendJson(res, 200, { funnel: store.funnel({ since }) });
        return;
      }

      if (pathname === '/api/eval') {
        const since = url.searchParams.get('since') || undefined;
        if (typeof evalRuns !== 'function') {
          sendJson(res, 200, { available: false, reason: 'eval is not wired into this viewer' });
          return;
        }
        const result = evalRuns({ store, since });
        Promise.resolve(result)
          .then((value) => sendJson(res, 200, { available: true, eval: value }))
          .catch((err) => sendJson(res, 500, { error: 'eval-failed', message: String(err?.message || err) }));
        return;
      }

      if (pathname === '/api/doctor') {
        if (typeof arming !== 'function') {
          sendJson(res, 200, { available: false, reason: 'arming report is not wired into this viewer' });
          return;
        }
        sendJson(res, 200, { available: true, arming: arming(), dbPath: store.path ?? config?.dbPath ?? null });
        return;
      }

      const receiptMatch = /^\/api\/receipt\/([^/]+)$/.exec(pathname);
      if (receiptMatch) {
        const runId = decodeURIComponent(receiptMatch[1]);
        const run = store.getRun(runId);
        if (!run) {
          notFound(res, `run ${runId}`);
          return;
        }
        if (!run.receipt) {
          sendJson(res, 200, { runId, receipt: null, verification: null });
          return;
        }
        let verification = null;
        if (typeof verifyReceipt === 'function') {
          try {
            verification = verifyReceipt(run.receipt);
          } catch (err) {
            verification = { ok: false, reason: String(err?.message || err) };
          }
        }
        Promise.resolve(verification)
          .then((value) => sendJson(res, 200, { runId, receipt: run.receipt, verification: value }))
          .catch((err) =>
            sendJson(res, 200, {
              runId,
              receipt: run.receipt,
              verification: { ok: false, reason: String(err?.message || err) },
            }),
          );
        return;
      }

      const streamMatch = /^\/api\/runs\/([^/]+)\/stream$/.exec(pathname);
      if (streamMatch) {
        const runId = decodeURIComponent(streamMatch[1]);
        if (!store.getRun(runId)) {
          notFound(res, `run ${runId}`);
          return;
        }
        openStream(req, res, runId);
        return;
      }

      const eventsMatch = /^\/api\/runs\/([^/]+)\/events$/.exec(pathname);
      if (eventsMatch) {
        const runId = decodeURIComponent(eventsMatch[1]);
        if (!store.getRun(runId)) {
          notFound(res, `run ${runId}`);
          return;
        }
        const after = intParam(url.searchParams.get('after'), 0);
        const limit = intParam(url.searchParams.get('limit'), 500);
        sendJson(res, 200, {
          runId,
          after,
          events: store.eventsSince(runId, Math.max(0, after), Math.max(1, limit)),
        });
        return;
      }

      const runMatch = /^\/api\/runs\/([^/]+)$/.exec(pathname);
      if (runMatch) {
        const runId = decodeURIComponent(runMatch[1]);
        const detail = runDetail(store, runId);
        if (!detail) {
          notFound(res, `run ${runId}`);
          return;
        }
        sendJson(res, 200, detail);
        return;
      }

      notFound(res, pathname);
    } catch (err) {
      sendJson(res, 500, {
        error: err?.code || 'viewer-error',
        message: String(err?.message || err),
      });
    }
  });

  server.on('connection', (socket) => {
    sockets.add(socket);
    socket.on('close', () => sockets.delete(socket));
  });

  return {
    server,
    host,
    get url() {
      return listeningPort === null ? null : formatUrl(host, listeningPort);
    },
    get port() {
      return listeningPort;
    },
    /** @returns {Promise<{port:number, url:string}>} */
    listen() {
      return new Promise((resolve, reject) => {
        const onError = (err) => {
          server.removeListener('listening', onListening);
          reject(err);
        };
        const onListening = () => {
          server.removeListener('error', onError);
          listeningPort = server.address().port;
          resolve({ port: listeningPort, url: formatUrl(host, listeningPort) });
        };
        server.once('error', onError);
        server.once('listening', onListening);
        server.listen(port, host);
      });
    },
    /** @returns {Promise<void>} */
    close() {
      return new Promise((resolve) => {
        for (const entry of [...streams]) closeStream(entry);
        server.close(() => resolve());
        for (const socket of [...sockets]) {
          try {
            socket.destroy();
          } catch {
            // socket already torn down
          }
        }
        sockets.clear();
      });
    },
  };
}

/** `http://host:port`, bracketing IPv6 literals. */
export function formatUrl(host, port) {
  const h = String(host);
  const shown = h.includes(':') && !h.startsWith('[') ? `[${h}]` : h;
  return `http://${shown}:${port}`;
}

export default createViewerServer;
