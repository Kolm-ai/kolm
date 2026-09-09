// kolm code — HTTP client for the Cloudflare Sandbox worker (owner B,
// spec sections 4 and 8). The worker lives in `deploy/kolm-code-sandbox`.
//
// Same worktree interface as the local driver; `path` is the container path the
// worker returns. `exec` responses are buffered (streaming is not required).
//
//   const driver = createCloudflareSandboxDriver(config, { fetchImpl });
//   const wt = await driver.create({ repoUrl, baseRef, runId, candidateId, gitToken });
//
// Arming: `config.sandboxUrl` + `config.sandboxToken` (or the same names in
// `opts`). An unarmed driver throws `LadderError('not-armed')` when USED, never
// at construction, so `armingReport` can describe it without tripping it.

import { LadderError } from '../errors.js';

export function createCloudflareSandboxDriver(config = {}, opts = {}) {
  const rawUrl = opts.sandboxUrl ?? config.sandboxUrl ?? null;
  const token = opts.sandboxToken ?? config.sandboxToken ?? null;
  const fetchImpl = opts.fetchImpl ?? globalThis.fetch;
  const defaultTimeoutMs = opts.timeoutMs ?? config.stageTimeoutMs ?? 300000;
  const baseUrl = typeof rawUrl === 'string' ? rawUrl.replace(/\/+$/, '') : null;
  const armed = Boolean(baseUrl && token);

  function requireArmed() {
    if (armed) return;
    throw new LadderError(
      'not-armed',
      'cloudflare worktree driver is not armed: set KOLM_CODE_SANDBOX_URL and KOLM_CODE_SANDBOX_TOKEN',
      { hasUrl: Boolean(baseUrl), hasToken: Boolean(token) },
    );
  }

  function buildUrl(pathname, query) {
    let url = `${baseUrl}${pathname}`;
    const params = new URLSearchParams();
    for (const [k, v] of Object.entries(query ?? {})) {
      if (v === undefined || v === null) continue;
      params.set(k, String(v));
    }
    const qs = params.toString();
    if (qs) url += `?${qs}`;
    return url;
  }

  async function request(method, pathname, { body, query, allow404 = false } = {}) {
    requireArmed();
    if (typeof fetchImpl !== 'function') {
      throw new LadderError('not-armed', 'no fetch implementation available for the cloudflare driver', {});
    }
    const url = buildUrl(pathname, query);
    const headers = { authorization: `Bearer ${token}`, accept: 'application/json' };
    if (body !== undefined) headers['content-type'] = 'application/json';
    let response;
    try {
      response = await fetchImpl(url, {
        method,
        headers,
        body: body === undefined ? undefined : JSON.stringify(body),
      });
    } catch (err) {
      throw new LadderError('sandbox-http', `${method} ${pathname} failed: ${err?.message ?? err}`, {
        status: 0,
        body: null,
      });
    }
    const text = typeof response.text === 'function' ? await response.text() : '';
    let parsed = null;
    if (text) {
      try { parsed = JSON.parse(text); } catch { parsed = null; }
    }
    if (allow404 && response.status === 404) return null;
    const ok = response.ok ?? (response.status >= 200 && response.status < 300);
    if (!ok) {
      throw new LadderError(
        'sandbox-http',
        `${method} ${pathname} -> ${response.status}`,
        { status: response.status, body: parsed ?? text },
      );
    }
    return parsed;
  }

  const runBase = (runId) => `/v1/runs/${encodeURIComponent(runId)}`;
  const wtBase = (runId, id) => `${runBase(runId)}/worktrees/${encodeURIComponent(id)}`;

  function makeWorktree({ runId, candidateId, id, containerPath, baseShaValue, repoUrl }) {
    const wt = {
      id,
      kind: 'cloudflare',
      path: containerPath,
      runId,
      candidateId,
      repoUrl,
      branch: `kolm/code/${runId}/${candidateId}`,

      async baseSha() {
        return baseShaValue;
      },

      async exec(cmd, execOpts = {}) {
        if (typeof cmd !== 'string' || cmd.trim() === '') {
          throw new LadderError('bad-config', 'exec requires a non-empty command string', { cmd });
        }
        const body = await request('POST', `${wtBase(runId, id)}/exec`, {
          body: {
            cmd,
            cwd: execOpts.cwd,
            timeoutMs: execOpts.timeoutMs ?? defaultTimeoutMs,
            env: execOpts.env,
            maxOutputBytes: execOpts.maxOutputBytes,
          },
        });
        return {
          stdout: body?.stdout ?? '',
          stderr: body?.stderr ?? '',
          exitCode: body?.exitCode ?? -1,
          durationMs: body?.durationMs ?? 0,
          timedOut: body?.timedOut ?? false,
          truncated: body?.truncated ?? false,
        };
      },

      async readFile(rel) {
        const body = await request('GET', `${wtBase(runId, id)}/file`, {
          query: { path: rel },
          allow404: true,
        });
        if (body === null) return null;
        return body.content ?? null;
      },

      async writeFile(rel, content) {
        return request('PUT', `${wtBase(runId, id)}/file`, {
          body: { path: rel, content: content ?? '' },
        });
      },

      async deleteFile(rel) {
        return request('DELETE', `${wtBase(runId, id)}/file`, { query: { path: rel } });
      },

      async listFiles({ glob, limit = 5000 } = {}) {
        const body = await request('GET', `${wtBase(runId, id)}/files`, { query: { glob, limit } });
        return body?.files ?? [];
      },

      async search(pattern, { glob, limit = 200 } = {}) {
        const body = await request('GET', `${wtBase(runId, id)}/search`, {
          query: { pattern, glob, limit },
        });
        return body?.hits ?? [];
      },

      async diff() {
        const body = await request('GET', `${wtBase(runId, id)}/diff`);
        return body?.diff ?? '';
      },

      async changedFiles() {
        const body = await request('GET', `${wtBase(runId, id)}/changed`);
        return body?.files ?? [];
      },

      async applyPatch(diff) {
        try {
          const body = await request('POST', `${wtBase(runId, id)}/patch`, { body: { diff } });
          return { ok: body?.ok !== false, error: body?.error };
        } catch (err) {
          if (err instanceof LadderError && err.code === 'sandbox-http' && err.detail?.status === 422) {
            return { ok: false, error: err.detail?.body?.error ?? err.message };
          }
          throw err;
        }
      },

      async commit(message) {
        const body = await request('POST', `${wtBase(runId, id)}/commit`, {
          body: { message: String(message ?? '') },
        });
        return { sha: body?.sha ?? null, branch: body?.branch ?? wt.branch };
      },

      async merge() {
        // Subtask integration (spec 4.4) needs a real three-way merge in the
        // container. The sandbox protocol has no endpoint for it yet, so this
        // fails loud rather than silently producing a wrong tree; the
        // orchestrator falls back to a single-unit run.
        throw new LadderError(
          'unsupported',
          'the cloudflare sandbox driver does not support merge yet; subtasks need the local git driver',
          { driver: 'cloudflare' },
        );
      },

      async destroy() {
        await request('DELETE', wtBase(runId, id));
        return { removed: true };
      },
    };
    return wt;
  }

  async function create({ repoUrl, repoPath, baseRef = 'HEAD', runId, candidateId, gitToken } = {}) {
    requireArmed();
    const url = repoUrl ?? repoPath;
    if (typeof url !== 'string' || url.length === 0) {
      throw new LadderError('bad-config', 'the cloudflare driver needs a repoUrl to clone', {
        repoUrl,
        repoPath,
      });
    }
    const body = await request('POST', `${runBase(runId)}/worktrees`, {
      body: { repoUrl: url, baseRef, candidateId, ...(gitToken ? { gitToken } : {}) },
    });
    return makeWorktree({
      runId,
      candidateId,
      id: body?.id ?? candidateId,
      containerPath: body?.path ?? null,
      baseShaValue: body?.baseSha ?? null,
      repoUrl: url,
    });
  }

  async function open({ runId, candidateId } = {}) {
    requireArmed();
    const id = candidateId;
    const body = await request('GET', wtBase(runId, id));
    return makeWorktree({
      runId,
      candidateId,
      id: body?.id ?? id,
      containerPath: body?.path ?? null,
      baseShaValue: body?.baseSha ?? null,
      repoUrl: body?.repoUrl ?? null,
    });
  }

  async function destroyRun(runId) {
    await request('DELETE', runBase(runId));
    return { removed: true };
  }

  return {
    kind: 'cloudflare',
    armed,
    sandboxUrl: baseUrl,
    create,
    open,
    destroyRun,
    /**
     * Not supported by the sandbox protocol; always throws `unsupported`.
     * @param {object} wt
     * @param {string} ref
     * @returns {Promise<never>}
     */
    merge: (wt, ref) => wt.merge(ref),
  };
}

export default createCloudflareSandboxDriver;
