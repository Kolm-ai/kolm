// kolm code — the coder tool loop (spec 6.2/6.3). Owner D.
//
// One candidate, one worktree, one bounded conversation. The loop is the same
// for every provider because the history is always Anthropic-shaped: an
// assistant turn carries `tool_use` blocks, and the next user turn carries one
// `tool_result` block per `tool_use`, keyed by `tool_use_id`. The
// OpenAI-compatible adapter translates that shape internally, so this file
// never branches on which model it is talking to.
//
// Everything here is bounded on purpose: turns, run output bytes, command
// prefixes, forbidden paths, and the run's cost ledger. A tool that refuses is
// NOT an exception — it returns the refusal text to the model as the tool
// result, because "you may not write that path" is information the coder can
// act on, while a thrown error just loses the candidate.

import { LadderError } from './errors.js';
import { matchGlobs } from './stages/l0-meta.js';
import { rubricToPrompt } from './rubric.js';
import { briefToPrompt } from './context/repo-brief.js';

const RUN_OUTPUT_CAP = 16 * 1024;
const READ_FILE_CAP = 64 * 1024;
const SEARCH_HIT_CAP = 60;
const LIST_FILE_CAP = 400;
const BRIEF_CAP = 8 * 1024;

/** Command prefixes a coder may `run` when the config does not override them. */
export const DEFAULT_ALLOWED_COMMANDS = Object.freeze([
  'npm', 'npx', 'node', 'pnpm', 'yarn',
  'pytest', 'python', 'go', 'cargo', 'make',
  'ls', 'cat', 'grep',
  'git diff', 'git status', 'git log',
]);

const SYSTEM = [
  'You are the kolm code coder. You change one repository, in one worktree, to satisfy one plan.',
  'Work through tools only: nothing you write in prose reaches the repository.',
  'Touch only what the task needs. A patch that edits unrelated files fails the review even if it works.',
  'Never leave a TODO, a stub, or a function that returns a canned value in place of the work.',
  'Read before you write; run the tests before you call finish.',
  'Call finish exactly once, with a summary of what you changed and why.',
].join(' ');

/** The tool set (spec 6.3), in Anthropic `input_schema` shape. */
export const CODER_TOOLS = Object.freeze([
  {
    name: 'read_file',
    description: 'Read one file from the worktree. Returns its full text, or a not-found note.',
    input_schema: {
      type: 'object',
      required: ['path'],
      properties: { path: { type: 'string', description: 'repo-relative path' } },
    },
  },
  {
    name: 'write_file',
    description: 'Create or overwrite one file with the exact content given. Write whole files, not fragments.',
    input_schema: {
      type: 'object',
      required: ['path', 'content'],
      properties: {
        path: { type: 'string', description: 'repo-relative path' },
        content: { type: 'string' },
      },
    },
  },
  {
    name: 'delete_file',
    description: 'Delete one file from the worktree.',
    input_schema: {
      type: 'object',
      required: ['path'],
      properties: { path: { type: 'string' } },
    },
  },
  {
    name: 'list_files',
    description: 'List repo-relative paths, optionally filtered by a glob such as src/**/*.js.',
    input_schema: {
      type: 'object',
      properties: { glob: { type: 'string' } },
    },
  },
  {
    name: 'search',
    description: 'Regex search across the worktree. Returns path:line:text hits.',
    input_schema: {
      type: 'object',
      required: ['pattern'],
      properties: {
        pattern: { type: 'string', description: 'JavaScript regular expression' },
        glob: { type: 'string' },
      },
    },
  },
  {
    name: 'run',
    description: 'Run one allow-listed shell command in the worktree (tests, typecheck, lint). Output is truncated.',
    input_schema: {
      type: 'object',
      required: ['cmd'],
      properties: { cmd: { type: 'string' } },
    },
  },
  {
    name: 'finish',
    description: 'Declare the work complete. Call this exactly once, last.',
    input_schema: {
      type: 'object',
      required: ['summary'],
      properties: { summary: { type: 'string' } },
    },
  },
]);

function capBytes(text, cap) {
  const src = String(text ?? '');
  const bytes = Buffer.byteLength(src, 'utf8');
  if (bytes <= cap) return src;
  const head = Buffer.from(src, 'utf8').subarray(0, Math.max(0, cap - 8)).toString('utf8');
  return `${head}\n[truncated ${bytes - Buffer.byteLength(head, 'utf8')} bytes]`;
}

/**
 * Is `cmd` allowed? Returns null when allowed, a reason string when not.
 * @param {string} cmd
 * @param {string[]} [allowed] prefixes
 * @returns {string|null}
 */
export function commandRefusal(cmd, allowed) {
  const list = Array.isArray(allowed) && allowed.length > 0 ? allowed : DEFAULT_ALLOWED_COMMANDS;
  const trimmed = String(cmd ?? '').trim();
  if (trimmed === '') return 'refused: empty command';
  // Chaining would let a denied command ride along behind an allowed one.
  if (/[;&|`$(){}<>]|\n/.test(trimmed)) {
    return `refused: command contains shell metacharacters; run one plain command at a time (allowed prefixes: ${list.join(', ')})`;
  }
  for (const prefix of list) {
    const p = String(prefix).trim();
    if (p === '') continue;
    if (trimmed === p || trimmed.startsWith(`${p} `)) return null;
  }
  return `refused: "${trimmed.split(/\s+/)[0]}" is not an allowed command prefix (allowed: ${list.join(', ')})`;
}

function buildSystem({ plan, rubric }) {
  const parts = [SYSTEM, ''];
  parts.push('PLAN');
  parts.push(`title: ${plan?.title ?? '(none)'}`);
  parts.push(`summary: ${plan?.summary ?? '(none)'}`);
  if (Array.isArray(plan?.acceptance) && plan.acceptance.length > 0) {
    parts.push('acceptance criteria (every one of these is checked later):');
    for (const a of plan.acceptance) parts.push(`  - ${a}`);
  }
  if (Array.isArray(plan?.files_expected) && plan.files_expected.length > 0) {
    parts.push(`files expected to change: ${plan.files_expected.join(', ')}`);
  }
  if (Array.isArray(plan?.forbidden_paths) && plan.forbidden_paths.length > 0) {
    parts.push(`forbidden paths (writes here are refused): ${plan.forbidden_paths.join(', ')}`);
  }
  if (plan?.test_command) parts.push(`test command: ${plan.test_command}`);
  if (rubric) {
    parts.push('');
    parts.push('YOUR PATCH WILL BE SCORED AGAINST THIS RUBRIC');
    parts.push(rubricToPrompt(rubric));
  }
  return parts.join('\n');
}

function buildOpening({ plan, brief, baseline, commands, feedback }) {
  const parts = [];
  if (feedback) {
    parts.push('REPAIR TURN — your previous attempt was rejected. Fix exactly this, then finish.');
    if (feedback.stage) parts.push(`rejected at: ${feedback.stage}`);
    for (const reason of (Array.isArray(feedback.reasons) ? feedback.reasons : []).slice(0, 12)) {
      parts.push(`  - ${reason}`);
    }
    for (const hit of (Array.isArray(feedback.hits) ? feedback.hits : []).slice(0, 12)) {
      if (!hit || typeof hit !== 'object') continue;
      parts.push(`  metric: ${hit.key} = ${hit.value} -> ${hit.band}`);
    }
    for (const failing of (Array.isArray(feedback.failing) ? feedback.failing : []).slice(0, 20)) {
      parts.push(`  failing: ${typeof failing === 'string' ? failing : JSON.stringify(failing)}`);
    }
    if (feedback.instructions) parts.push(`  instructions: ${feedback.instructions}`);
    parts.push('');
    parts.push('Keep everything that already works. Do not start over.');
    parts.push('');
  }
  parts.push('TASK');
  parts.push(plan?.summary ? String(plan.summary) : String(plan?.title ?? 'implement the plan'));
  parts.push('');
  parts.push(capBytes(briefToPrompt(brief), BRIEF_CAP));
  if (baseline) {
    parts.push('');
    parts.push('BASELINE (the repo BEFORE your change — do not "fix" these unless the task says so)');
    const tests = baseline.tests || {};
    parts.push(`  tests: passed=${tests.passed ?? '?'} failed=${tests.failed ?? '?'} exit=${tests.exitCode ?? '?'}`);
    for (const f of (Array.isArray(tests.failing) ? tests.failing : []).slice(0, 10)) {
      parts.push(`    already failing: ${typeof f === 'string' ? f : JSON.stringify(f)}`);
    }
  }
  const testCmd = plan?.test_command || commands?.test;
  if (testCmd) {
    parts.push('');
    parts.push(`Run \`${testCmd}\` before you finish.`);
  }
  parts.push('');
  parts.push('Start by reading the files you intend to change.');
  return parts.join('\n');
}

/** Execute one tool call; always resolves to a string for the model. */
async function executeTool({ call, wt, plan, config, signal }) {
  const input = call.input && typeof call.input === 'object' ? call.input : {};
  try {
    switch (call.name) {
      case 'read_file': {
        const text = await wt.readFile(String(input.path ?? ''));
        if (text === null || text === undefined) return `not found: ${input.path}`;
        return capBytes(text, READ_FILE_CAP);
      }
      case 'write_file': {
        const target = String(input.path ?? '');
        const forbidden = matchGlobs(target, plan?.forbidden_paths || []);
        if (forbidden) return `refused: ${target} matches the forbidden path ${forbidden}`;
        await wt.writeFile(target, String(input.content ?? ''));
        return `wrote ${target} (${Buffer.byteLength(String(input.content ?? ''), 'utf8')} bytes)`;
      }
      case 'delete_file': {
        const target = String(input.path ?? '');
        const forbidden = matchGlobs(target, plan?.forbidden_paths || []);
        if (forbidden) return `refused: ${target} matches the forbidden path ${forbidden}`;
        await wt.deleteFile(target);
        return `deleted ${target}`;
      }
      case 'list_files': {
        const files = await wt.listFiles({ glob: input.glob, limit: LIST_FILE_CAP + 1 });
        const shown = files.slice(0, LIST_FILE_CAP);
        const more = files.length > shown.length ? `\n[${files.length - shown.length} more not shown]` : '';
        return shown.length === 0 ? '(no files matched)' : `${shown.join('\n')}${more}`;
      }
      case 'search': {
        const hits = await wt.search(String(input.pattern ?? ''), { glob: input.glob, limit: SEARCH_HIT_CAP });
        if (hits.length === 0) return '(no matches)';
        return hits.map((h) => `${h.path}:${h.line}: ${String(h.text).slice(0, 200)}`).join('\n');
      }
      case 'run': {
        const cmd = String(input.cmd ?? '');
        const refusal = commandRefusal(cmd, config?.coderAllowedCommands);
        if (refusal) return refusal;
        const res = await wt.exec(cmd, { timeoutMs: 120000, signal });
        const body = [
          `exit code: ${res?.exitCode ?? '?'}`,
          String(res?.stdout ?? ''),
          String(res?.stderr ?? ''),
        ].join('\n');
        return capBytes(body, RUN_OUTPUT_CAP);
      }
      case 'finish':
        return 'acknowledged';
      default:
        return `unknown tool: ${call.name}`;
    }
  } catch (err) {
    // path-escape and friends are answers, not crashes: hand the message back.
    return `error: ${err?.message || String(err)}`;
  }
}

function overBudget(store, runId, config) {
  const cap = Number(config?.maxCostUsd);
  if (!Number.isFinite(cap) || cap <= 0) return false;
  if (!store || !runId || typeof store.costSummary !== 'function') return false;
  return Number(store.costSummary(runId).total_usd) >= cap;
}

/**
 * Run the coder loop for one candidate.
 *
 * @param {object} args
 * @param {string} args.runId
 * @param {string} args.candidateId
 * @param {object} args.wt worktree handle
 * @param {object} args.plan
 * @param {object} [args.brief]
 * @param {object} [args.baseline]
 * @param {object} [args.commands]
 * @param {object} [args.rubric]
 * @param {object} args.coder provider
 * @param {object} args.store
 * @param {object} [args.config]
 * @param {AbortSignal} [args.signal]
 * @param {number} [args.maxTurns] defaults to config.coderTurns
 * @param {object} [args.logger]
 * @param {object} [args.feedback] repair feedback (see runRepair)
 * @returns {Promise<{summary:string, turns:number, finished:boolean, reason:string,
 *   diff:string, changedFiles:Array, costUsd:number, model:string|null, toolCalls:number}>}
 */
export async function runCoder({
  runId,
  candidateId,
  wt,
  plan,
  brief,
  baseline,
  commands,
  rubric,
  coder,
  store,
  config = {},
  signal,
  maxTurns,
  logger,
  feedback = null,
} = {}) {
  if (!wt) throw new LadderError('bad-arg', 'runCoder needs a worktree');
  if (!coder || typeof coder.chat !== 'function') throw new LadderError('bad-arg', 'runCoder needs a coder provider');
  if (coder.armed === false) {
    throw new LadderError('not-armed', `the coder rung ${coder.id ?? '?'} is not armed`, { ref: coder.id ?? null });
  }

  const turnCap = Math.max(1, Math.trunc(Number(maxTurns ?? config.coderTurns ?? 24)));
  const system = buildSystem({ plan, rubric });
  const messages = [{ role: 'user', content: buildOpening({ plan, brief, baseline, commands, feedback }) }];

  let turns = 0;
  let toolCallCount = 0;
  let costUsd = 0;
  let summary = '';
  let finished = false;
  let reason = 'max-turns';
  let model = coder.id ?? null;
  let idleTurns = 0;

  for (let turn = 1; turn <= turnCap; turn += 1) {
    if (signal?.aborted) {
      reason = 'aborted';
      break;
    }
    if (overBudget(store, runId, config)) {
      reason = 'budget';
      break;
    }

    turns = turn;
    const reply = await coder.chat({
      system,
      messages,
      tools: CODER_TOOLS,
      maxTokens: Number(config.coderMaxTokens) > 0 ? Number(config.coderMaxTokens) : 8000,
      temperature: 0.2,
      cacheSystem: true,
      signal,
      timeoutMs: Number(config.stageTimeoutMs) > 0 ? Number(config.stageTimeoutMs) : 600000,
    });

    const turnCost = Number(reply?.costUsd) || 0;
    costUsd += turnCost;
    model = coder.id ?? reply?.model ?? model;
    const calls = Array.isArray(reply?.toolCalls) ? reply.toolCalls : [];
    if (store && runId) {
      store.addCost({
        runId,
        candidateId,
        stage: 'coder',
        provider: reply?.provider ?? coder.provider ?? null,
        model: reply?.model ?? coder.model ?? null,
        inputTokens: reply?.usage?.input ?? 0,
        outputTokens: reply?.usage?.output ?? 0,
        cacheRead: reply?.usage?.cacheRead ?? 0,
        costUsd: turnCost,
      });
      store.emit(runId, 'coder.turn', {
        candidateId,
        turn,
        toolCalls: calls.map((c) => c.name),
        costUsd: turnCost,
      });
    }

    if (reply?.stopReason === 'refusal') {
      reason = 'refusal';
      break;
    }

    if (calls.length === 0) {
      idleTurns += 1;
      if (idleTurns >= 2) {
        reason = 'no-tool-calls';
        break;
      }
      messages.push({ role: 'assistant', content: reply?.content ?? [{ type: 'text', text: reply?.text ?? '' }] });
      messages.push({
        role: 'user',
        content: 'You made no tool call. Use the tools to change the repository, then call finish.',
      });
      continue;
    }
    idleTurns = 0;
    toolCallCount += calls.length;

    messages.push({ role: 'assistant', content: reply.content });
    const results = [];
    let sawFinish = false;
    for (const call of calls) {
      if (call.name === 'finish') {
        sawFinish = true;
        summary = String(call.input?.summary ?? '').slice(0, 4000);
      }
      const text = await executeTool({ call, wt, plan, config, signal });
      results.push({ type: 'tool_result', tool_use_id: call.id, content: String(text) });
    }
    messages.push({ role: 'user', content: results });

    if (sawFinish) {
      finished = true;
      reason = 'finish';
      break;
    }
  }

  if (!finished && reason === 'max-turns') logger?.warn?.(`coder ${candidateId} hit the turn cap (${turnCap})`);

  const diff = await wt.diff();
  const changedFiles = await wt.changedFiles();
  if (!summary) {
    summary = finished
      ? 'coder finished without a summary'
      : `coder stopped: ${reason} (${changedFiles.length} file(s) changed)`;
  }

  if (store && runId && candidateId) {
    store.upsertCandidate({
      runId,
      candidateId,
      diffText: diff,
      summary,
      status: finished ? 'coded' : `stopped-${reason}`,
    });
  }

  return {
    summary,
    turns,
    finished,
    reason,
    diff,
    changedFiles,
    costUsd,
    model,
    toolCalls: toolCallCount,
  };
}

/**
 * One bounded repair pass over an existing worktree, with the failure that
 * rejected it injected as the opening message.
 *
 * @param {object} args same as runCoder, plus:
 * @param {{stage?:string, reasons?:string[], hits?:Array, failing?:Array, instructions?:string}} args.feedback
 * @returns {Promise<object>} same shape as runCoder
 */
export async function runRepair(args = {}) {
  if (!args.feedback) throw new LadderError('bad-arg', 'runRepair needs feedback');
  return runCoder({
    ...args,
    maxTurns: args.maxTurns ?? args.config?.repairTurns ?? 2,
  });
}

export default { runCoder, runRepair, CODER_TOOLS, DEFAULT_ALLOWED_COMMANDS, commandRefusal };
