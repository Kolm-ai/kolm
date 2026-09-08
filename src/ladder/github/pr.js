// kolm code — push a winner branch and open its pull request (owner F, spec 9).
//
// The token comes from `resolveGitHubToken` (app-auth.js) and is injected into
// git through `-c http.extraheader=...` for one invocation only: it is never
// written into a remote URL, a config file or a credential helper, and it is
// redacted out of any git stderr that reaches a LadderError.

import { execFileSync } from 'node:child_process';

import { LadderError } from '../errors.js';
import { GITHUB_API_BASE, GITHUB_API_VERSION, USER_AGENT, redactSecrets } from './app-auth.js';

const PUSH_TIMEOUT_MS = 120_000;
const GIT_TIMEOUT_MS = 15_000;
const BODY_EXCERPT = 300;

function apiHeaders(token) {
  return {
    authorization: `Bearer ${token}`,
    accept: 'application/vnd.github+json',
    'x-github-api-version': GITHUB_API_VERSION,
    'user-agent': USER_AGENT,
    'content-type': 'application/json',
  };
}

async function bodyText(response) {
  try {
    return typeof response?.text === 'function' ? await response.text() : '';
  } catch {
    return '';
  }
}

function base(apiBase) {
  return String(apiBase || GITHUB_API_BASE).replace(/\/+$/, '');
}

function defaultExec(file, args, options) {
  return execFileSync(file, args, options);
}

/**
 * Parses `owner`/`repo` out of the three remote URL spellings GitHub hands out.
 * @param {string} remoteUrl
 * @returns {{owner: string, repo: string}|null}
 */
export function remoteToOwnerRepo(remoteUrl) {
  const raw = typeof remoteUrl === 'string' ? remoteUrl.trim() : '';
  if (raw === '') return null;
  const patterns = [
    /^https?:\/\/[^/]*github\.com\/([\w.-]+)\/([\w.-]+?)(?:\.git)?\/?$/i,
    /^ssh:\/\/(?:[^@/]+@)?[^/]*github\.com(?::\d+)?\/([\w.-]+)\/([\w.-]+?)(?:\.git)?\/?$/i,
    /^(?:[^@\s]+@)?[^:/\s]*github\.com:([\w.-]+)\/([\w.-]+?)(?:\.git)?\/?$/i,
  ];
  for (const pattern of patterns) {
    const match = pattern.exec(raw);
    if (match) {
      const owner = match[1];
      const repo = match[2];
      if (owner && repo && repo !== '.' && repo !== '..') return { owner, repo };
    }
  }
  return null;
}

/**
 * `git remote get-url <remote>` inside `repoPath`.
 * @param {object} opts
 * @returns {string|null} the remote URL, or null when git cannot answer
 */
export function detectRemote({ repoPath, remote = 'origin', execImpl } = {}) {
  const run = typeof execImpl === 'function' ? execImpl : defaultExec;
  try {
    const out = run('git', ['remote', 'get-url', remote], {
      cwd: repoPath,
      encoding: 'utf8',
      timeout: GIT_TIMEOUT_MS,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    const url = typeof out === 'string' ? out.trim() : String(out ?? '').trim();
    return url === '' ? null : url;
  } catch {
    return null;
  }
}

/**
 * Pushes `branch` to `remote` with the token carried in a one-shot
 * `http.extraheader`, the pattern actions/checkout uses for App tokens.
 * @param {object} opts
 * @returns {{pushed: true, remote: string, branch: string}}
 */
export function pushBranch({ repoPath, branch, remote = 'origin', token, execImpl, force = false } = {}) {
  if (!repoPath) throw new LadderError('git-push', 'repoPath is required to push a branch');
  if (!branch) throw new LadderError('git-push', 'branch is required to push a branch');
  const run = typeof execImpl === 'function' ? execImpl : defaultExec;

  const basic = token
    ? Buffer.from(`x-access-token:${token}`, 'utf8').toString('base64')
    : null;
  const args = [];
  if (basic) args.push('-c', `http.extraheader=AUTHORIZATION: basic ${basic}`);
  args.push('push');
  if (force) args.push('--force-with-lease');
  args.push(remote, `${branch}:refs/heads/${branch}`);

  try {
    run('git', args, {
      cwd: repoPath,
      encoding: 'utf8',
      timeout: PUSH_TIMEOUT_MS,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
  } catch (err) {
    const secrets = [token, basic].filter((v) => typeof v === 'string' && v.length > 0);
    const raw = [err?.stderr, err?.stdout, err?.message]
      .map((part) => (typeof part === 'string' ? part : part ? String(part) : ''))
      .join('\n')
      .trim();
    throw new LadderError('git-push', `git push failed for ${branch} -> ${remote}`, {
      remote,
      branch,
      status: typeof err?.status === 'number' ? err.status : null,
      stderr: redactSecrets(raw, secrets).slice(0, 2000),
    });
  }
  return { pushed: true, remote, branch };
}

/**
 * Creates a pull request, or returns the open one GitHub already has for this
 * head branch (GitHub answers 422 "A pull request already exists").
 * @param {object} opts
 * @returns {Promise<{number: number, url: string, htmlUrl: string, created: boolean}>}
 */
export async function openPullRequest({
  owner,
  repo,
  head,
  base: baseBranch = 'main',
  title,
  body = '',
  draft = true,
  token,
  fetchImpl = fetch,
  apiBase = GITHUB_API_BASE,
}) {
  if (!owner || !repo) {
    throw new LadderError('github-pr', 'owner and repo are required to open a pull request', {
      owner: owner ?? null,
      repo: repo ?? null,
    });
  }
  if (!head) throw new LadderError('github-pr', 'head branch is required to open a pull request');
  if (!token) {
    throw new LadderError('not-armed', 'No GitHub token is armed; `kolm code doctor` lists the sources');
  }

  const root = `${base(apiBase)}/repos/${owner}/${repo}`;
  let response;
  try {
    response = await fetchImpl(`${root}/pulls`, {
      method: 'POST',
      headers: apiHeaders(token),
      body: JSON.stringify({ title, head, base: baseBranch, body, draft: Boolean(draft) }),
    });
  } catch (err) {
    throw new LadderError('github-pr', 'GitHub pull request request failed', {
      reason: redactSecrets(String(err?.message || err?.code || 'network-error'), [token]),
    });
  }

  const status = Number(response?.status ?? 0);
  const text = await bodyText(response);
  if (response?.ok) {
    const payload = safeJson(text);
    return {
      number: payload?.number ?? null,
      url: payload?.url ?? null,
      htmlUrl: payload?.html_url ?? null,
      created: true,
    };
  }

  if (status === 422 && /a pull request already exists/i.test(text)) {
    const existing = await findOpenPullRequest({
      owner, repo, head, token, fetchImpl, apiBase,
    });
    if (existing) return existing;
  }

  throw new LadderError('github-pr', `GitHub pull request creation failed: ${status}`, {
    status,
    body: redactSecrets(text.slice(0, BODY_EXCERPT), [token]),
  });
}

async function findOpenPullRequest({ owner, repo, head, token, fetchImpl, apiBase }) {
  const qualified = String(head).includes(':') ? String(head) : `${owner}:${head}`;
  const url = `${base(apiBase)}/repos/${owner}/${repo}/pulls?head=${encodeURIComponent(qualified)}&state=open`;
  let response;
  try {
    response = await fetchImpl(url, { method: 'GET', headers: apiHeaders(token) });
  } catch {
    return null;
  }
  if (!response?.ok) return null;
  const list = safeJson(await bodyText(response));
  const first = Array.isArray(list) ? list[0] : null;
  if (!first) return null;
  return {
    number: first.number ?? null,
    url: first.url ?? null,
    htmlUrl: first.html_url ?? null,
    created: false,
  };
}

function safeJson(text) {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

function stageRows(receipt, runSummary) {
  const source = receipt?.stages ?? runSummary?.stages ?? receipt?.candidates?.[0]?.stages ?? null;
  const rows = [];
  if (Array.isArray(source)) {
    for (const entry of source) {
      if (!entry) continue;
      if (typeof entry === 'string') {
        rows.push({ stage: entry, decision: '', reasons: [] });
        continue;
      }
      rows.push({
        stage: entry.stage ?? entry.id ?? entry.name ?? 'stage',
        decision: entry.decision ?? entry.verdict ?? '',
        reasons: normalizeReasons(entry.reasons ?? entry.reason),
      });
    }
  } else if (source && typeof source === 'object') {
    for (const [stage, entry] of Object.entries(source)) {
      if (entry && typeof entry === 'object') {
        rows.push({
          stage,
          decision: entry.decision ?? entry.verdict ?? '',
          reasons: normalizeReasons(entry.reasons ?? entry.reason),
        });
      } else {
        rows.push({ stage, decision: String(entry ?? ''), reasons: [] });
      }
    }
  }
  return rows;
}

function normalizeReasons(value) {
  if (Array.isArray(value)) return value.map((v) => String(v));
  if (typeof value === 'string' && value !== '') return [value];
  return [];
}

function cell(value) {
  return String(value ?? '').replace(/\|/g, '\\|').replace(/\r?\n/g, ' ');
}

function firstString(...values) {
  for (const value of values) {
    if (typeof value === 'string' && value.trim() !== '') return value.trim();
  }
  return null;
}

/**
 * Markdown body for the pull request: what ran, what the cascade decided, the
 * receipt itself, and how to verify it offline. Never contains a credential.
 * @param {object} opts
 * @param {object} [opts.receipt] owner D's `kolm-code-receipt-v1`
 * @param {object} [opts.runSummary]
 * @param {string} [opts.receiptPath] path printed in the verify line
 * @returns {string}
 */
export function buildPrBody({ receipt = null, runSummary = null, receiptPath } = {}) {
  const runId = firstString(receipt?.run_id, receipt?.runId, runSummary?.runId, runSummary?.run_id);
  const title = firstString(
    receipt?.task?.title,
    receipt?.task?.text,
    typeof receipt?.task === 'string' ? receipt.task : null,
    runSummary?.task?.title,
    typeof runSummary?.task === 'string' ? runSummary.task : null,
    runSummary?.title,
  ) || 'kolm code run';

  const winner = firstString(
    receipt?.winner?.candidate,
    receipt?.winner?.id,
    typeof receipt?.winner === 'string' ? receipt.winner : null,
    runSummary?.winner?.candidate,
    typeof runSummary?.winner === 'string' ? runSummary.winner : null,
  );
  const branch = firstString(receipt?.winner?.branch, runSummary?.branch);
  const baseSha = firstString(receipt?.base_sha, receipt?.repo?.base_sha, runSummary?.baseSha);

  const costValue =
    receipt?.cost?.total_usd ?? receipt?.cost?.totalUsd ??
    (typeof receipt?.cost === 'number' ? receipt.cost : null) ??
    runSummary?.costUsd ?? runSummary?.cost?.total_usd ?? null;
  const cost = typeof costValue === 'number' ? `$${costValue.toFixed(4)}` : 'n/a';

  const verdict = firstString(receipt?.verdict, receipt?.judge?.verdict, runSummary?.verdict);

  const lines = [];
  lines.push(`## kolm code: ${title}`);
  lines.push('');
  lines.push(`- task: ${title}`);
  if (winner) lines.push(`- winner candidate: ${winner}`);
  if (branch) lines.push(`- branch: ${branch}`);
  if (baseSha) lines.push(`- base sha: ${baseSha}`);
  if (verdict) lines.push(`- verdict: ${verdict}`);
  lines.push(`- cost: ${cost}`);
  if (runId) lines.push(`- run id: ${runId}`);
  lines.push('');

  const rows = stageRows(receipt, runSummary);
  lines.push('### Cascade funnel');
  lines.push('');
  lines.push('| stage | decision | reasons |');
  lines.push('| --- | --- | --- |');
  if (rows.length === 0) {
    lines.push('| (no stage results recorded) | | |');
  } else {
    for (const row of rows) {
      lines.push(`| ${cell(row.stage)} | ${cell(row.decision)} | ${cell(row.reasons.join('; '))} |`);
    }
  }
  lines.push('');

  lines.push('### Receipt');
  lines.push('');
  lines.push('```json');
  lines.push(stringifyReceipt(receipt));
  lines.push('```');
  lines.push('');
  lines.push(`Verify: kolm code receipt verify ${receiptPath || defaultReceiptPath(runId)}`);
  lines.push('');
  lines.push('Generated by kolm code');

  return redactSecrets(lines.join('\n'));
}

function defaultReceiptPath(runId) {
  return runId ? `~/.kolm/code/runs/${runId}/receipt.json` : 'receipt.json';
}

function stringifyReceipt(receipt) {
  if (!receipt) return '{}';
  try {
    return JSON.stringify(receipt, null, 2);
  } catch {
    return '{}';
  }
}

/**
 * Detect owner/repo (when not given) -> push the branch -> open the PR.
 * @param {object} opts
 * @returns {Promise<{number: number|null, htmlUrl: string|null, url: string|null, pushed: boolean, created: boolean, owner: string, repo: string, branch: string}>}
 */
export async function openPrForRun({
  repoPath,
  branch,
  base: baseBranch = 'main',
  title,
  body = '',
  token,
  owner,
  repo,
  draft = true,
  remote = 'origin',
  fetchImpl = fetch,
  execImpl,
  apiBase = GITHUB_API_BASE,
}) {
  let target = owner && repo ? { owner, repo } : null;
  if (!target) {
    const remoteUrl = detectRemote({ repoPath, remote, execImpl });
    target = remoteToOwnerRepo(remoteUrl || '');
    if (!target) {
      throw new LadderError('github-pr', `Could not determine owner/repo from remote "${remote}"`, {
        repoPath: repoPath ?? null,
        remote,
        remoteUrl: remoteUrl ?? null,
      });
    }
  }

  const push = pushBranch({ repoPath, branch, remote, token, execImpl });
  const pr = await openPullRequest({
    owner: target.owner,
    repo: target.repo,
    head: branch,
    base: baseBranch,
    title: title || branch,
    body,
    draft,
    token,
    fetchImpl,
    apiBase,
  });

  return {
    number: pr.number,
    url: pr.url,
    htmlUrl: pr.htmlUrl,
    pushed: push.pushed,
    created: pr.created,
    owner: target.owner,
    repo: target.repo,
    branch,
  };
}
