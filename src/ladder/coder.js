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
import { modelTimeoutMs } from './config.js';
import { matchGlobs } from './stages/l0-meta.js';
import { rubricToPrompt } from './rubric.js';
import { briefToPrompt } from './context/repo-brief.js';
import { allowedForShell } from './worktree/local-git.js';

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

const TOOL_GUIDANCE = [
  'TOOL DISCIPLINE',
  '- To change part of a file that already exists, use edit_file: it replaces one exact,'
    + ' unique old_string with new_string and leaves the rest of the file untouched.',
  '- Use write_file only for a NEW file or a deliberate full rewrite. Rewriting a large file'
    + ' to change three lines is how patches acquire damage.',
  '- Never inspect or edit files through run. read_file, search and list_files exist for that,'
    + ' they cost nothing, and they cannot be defeated by quoting rules.',
  '- run is for tests, typecheck and lint. Nothing else.',
  '- Diagnosis is not the task. If you have read the relevant file, make the change; you can'
    + ' always run the tests afterwards and adjust.',
].join('\n');

const TURN_INPUT_CAP = 200;
const TURN_RESULT_CAP = 300;
const TURN_DETAIL_CAP = 4000;

/** Wall-clock budget for one `run` tool call. */
const RUN_TIMEOUT_MS = 120000;

/** How many consecutive turns without a file change trigger a nudge. */
export const NO_PROGRESS_WINDOW = 6;
/** Tools that count as making progress. */
const WRITING_TOOLS = new Set(['write_file', 'edit_file', 'delete_file']);

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
    name: 'edit_file',
    description:
      'Replace one exact occurrence of old_string with new_string in an existing file. '
      + 'old_string must appear exactly once; include enough surrounding context to make it unique. '
      + 'This is the tool to reach for when changing a file that already exists.',
    input_schema: {
      type: 'object',
      required: ['path', 'old_string', 'new_string'],
      properties: {
        path: { type: 'string', description: 'repo-relative path' },
        old_string: { type: 'string', description: 'exact text to replace; must be unique in the file' },
        new_string: { type: 'string', description: 'text to put in its place' },
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

/** Count non-overlapping occurrences of `needle` in `haystack`. */
function countOccurrences(haystack, needle) {
  if (needle === '') return 0;
  let count = 0;
  let at = haystack.indexOf(needle);
  while (at !== -1) {
    count += 1;
    at = haystack.indexOf(needle, at + needle.length);
  }
  return count;
}

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

function buildSystem({ plan, rubric, shell, allowedCommands }) {
  const parts = [SYSTEM, ''];
  parts.push(TOOL_GUIDANCE);
  parts.push('');
  parts.push('ENVIRONMENT');
  parts.push(`platform: ${shell?.platform ?? process.platform}`);
  parts.push(`run executes through: ${shellLabel(shell)}`);
  if (shell && shell.posix === false) {
    parts.push('There is no POSIX shell here: ls, cat, grep, pipes and globbing are NOT available to run.');
  }
  parts.push('read_file, list_files and search never go through the shell, so they work the same everywhere.');
  if (Array.isArray(allowedCommands) && allowedCommands.length > 0) {
    parts.push(`allowed run prefixes: ${allowedCommands.join(', ')}`);
  }
  parts.push('');
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

/** The 200-char identifying argument of a tool call (never file content). */
function toolInputSummary(call) {
  const input = call?.input && typeof call.input === 'object' ? call.input : {};
  const value = input.path ?? input.cmd ?? input.pattern ?? input.glob ?? input.summary ?? '';
  return String(value).slice(0, TURN_INPUT_CAP);
}

/**
 * Bounded per-tool-call detail for the `coder.turn` event: what was called,
 * which path/command it named, and the head of what came back. Never the
 * content a write_file carried.
 * @param {Array<{name:string, input?:object}>} calls
 * @param {string[]} results tool result strings, index-aligned with `calls`
 * @returns {Array<{name:string, input:string, result:string}>}
 */
export function turnDetail(calls, results) {
  const out = [];
  for (let i = 0; i < calls.length; i += 1) {
    out.push({
      name: String(calls[i]?.name ?? '?'),
      input: toolInputSummary(calls[i]),
      result: String(results[i] ?? '').slice(0, TURN_RESULT_CAP),
    });
  }
  // Keep the whole event comfortably under 4 KB.
  while (out.length > 1 && JSON.stringify(out).length > TURN_DETAIL_CAP) out.pop();
  return out;
}

/** Human-readable name of the shell `run` uses. */
function shellLabel(shell) {
  if (!shell) return 'the platform default shell';
  if (shell.kind === 'bash') return `bash (${shell.path ?? 'bash'})`;
  if (shell.kind === 'cmd') return 'cmd.exe';
  return shell.path ? `sh (${shell.path})` : '/bin/sh';
}

/**
 * The `run` allow-list for this worktree: the config's list (or the default),
 * minus anything the resolved shell cannot run.
 * @param {object} [wt]
 * @param {object} [config]
 * @returns {string[]}
 */
export function allowedCommandsFor(wt, config) {
  const configured = Array.isArray(config?.coderAllowedCommands) && config.coderAllowedCommands.length > 0
    ? config.coderAllowedCommands
    : DEFAULT_ALLOWED_COMMANDS;
  return allowedForShell([...configured], wt?.shell);
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
      case 'edit_file': {
        const target = String(input.path ?? '');
        const forbidden = matchGlobs(target, plan?.forbidden_paths || []);
        if (forbidden) return `refused: ${target} matches the forbidden path ${forbidden}`;
        const oldString = String(input.old_string ?? '');
        const newString = String(input.new_string ?? '');
        if (oldString === '') return `refused: edit_file needs a non-empty old_string for ${target}`;
        const before = await wt.readFile(target);
        if (before === null || before === undefined) return `not found: ${target}`;
        // Models emit LF; a CRLF file must still match, and the edit must keep
        // the file's own line endings (live run r_20260909_f0586198 burned three
        // turns on "0 occurrences" against a CRLF fixture).
        let needle = oldString;
        let replacement = newString;
        let count = countOccurrences(before, needle);
        if (count === 0 && before.includes('\r\n') && !oldString.includes('\r')) {
          needle = oldString.replace(/\n/g, '\r\n');
          replacement = newString.replace(/\r?\n/g, '\r\n');
          count = countOccurrences(before, needle);
        }
        if (count === 0) {
          return `refused: old_string is not present in ${target} (0 occurrences). `
            + 'Read the file again and copy the exact text, including whitespace.';
        }
        if (count > 1) {
          return `refused: old_string appears ${count} times in ${target}; it must appear exactly once. `
            + 'Add surrounding lines to make it unique.';
        }
        if (needle === replacement) return `no change: old_string and new_string are identical in ${target}`;
        const at = before.indexOf(needle);
        const after = before.slice(0, at) + replacement + before.slice(at + needle.length);
        await wt.writeFile(target, after);
        return `edited ${target} (1 replacement, ${after.length - before.length >= 0 ? '+' : ''}`
          + `${after.length - before.length} chars)`;
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
        const refusal = commandRefusal(cmd, allowedCommandsFor(wt, config));
        if (refusal) return refusal;
        const res = await wt.exec(cmd, { timeoutMs: RUN_TIMEOUT_MS, signal });
        const body = [
          res?.timedOut
            ? `timed out after ${RUN_TIMEOUT_MS} ms and was killed (exit code: ${res?.exitCode ?? '?'}); do not rerun it`
            : `exit code: ${res?.exitCode ?? '?'}`,
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
  const system = buildSystem({
    plan,
    rubric,
    shell: wt.shell,
    allowedCommands: allowedCommandsFor(wt, config),
  });
  const messages = [{ role: 'user', content: buildOpening({ plan, brief, baseline, commands, feedback }) }];

  let turns = 0;
  let toolCallCount = 0;
  let costUsd = 0;
  let summary = '';
  let finished = false;
  let reason = 'max-turns';
  let model = coder.id ?? null;
  let idleTurns = 0;
  // Progress watchdog: a coder that reads and greps forever is a coder that
  // never lands a patch. Count turns whose tool calls changed no file.
  let turnsWithoutChange = 0;
  let nudged = 0;

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
      timeoutMs: modelTimeoutMs(config),
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
    }
    // Emitted once per turn; when the turn ran tools it carries their bounded
    // detail, which is the difference between a replayable run and a list of
    // tool names.
    const emitTurn = (detail) => {
      if (!store || !runId) return;
      store.emit(runId, 'coder.turn', {
        candidateId,
        turn,
        toolCalls: calls.map((c) => c.name),
        costUsd: turnCost,
        detail: detail ?? [],
      });
    };

    if (reply?.stopReason === 'refusal') {
      emitTurn(null);
      reason = 'refusal';
      break;
    }

    if (calls.length === 0) {
      emitTurn(null);
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
    const resultTexts = [];
    for (const call of calls) {
      if (call.name === 'finish') {
        sawFinish = true;
        summary = String(call.input?.summary ?? '').slice(0, 4000);
      }
      const text = await executeTool({ call, wt, plan, config, signal });
      resultTexts.push(String(text));
      results.push({ type: 'tool_result', tool_use_id: call.id, content: String(text) });
    }
    messages.push({ role: 'user', content: results });
    emitTurn(turnDetail(calls, resultTexts));

    if (sawFinish) {
      finished = true;
      reason = 'finish';
      break;
    }

    // --- progress watchdog ---------------------------------------------
    turnsWithoutChange = calls.some((c) => WRITING_TOOLS.has(c.name)) ? 0 : turnsWithoutChange + 1;
    if (turnsWithoutChange >= NO_PROGRESS_WINDOW) {
      if (nudged === 0) {
        nudged = 1;
        turnsWithoutChange = 0;
        store?.emit?.(runId, 'coder.nudge', {
          candidateId,
          turn,
          turnsWithoutChange: NO_PROGRESS_WINDOW,
        });
        messages.push({
          role: 'user',
          content: `${NO_PROGRESS_WINDOW} turns without a file change; make the change now with `
            + 'edit_file or write_file, then run the tests.',
        });
        continue;
      }
      store?.emit?.(runId, 'coder.no-progress', {
        candidateId,
        turn,
        turnsWithoutChange: NO_PROGRESS_WINDOW,
      });
      reason = 'no-progress';
      break;
    }
  }

  if (!finished && reason === 'max-turns') logger?.warn?.(`coder ${candidateId} hit the turn cap (${turnCap})`);
  if (reason === 'no-progress') {
    logger?.warn?.(`coder ${candidateId} stopped: ${NO_PROGRESS_WINDOW * 2} turns without a file change`);
  }

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
    maxTurns: args.maxTurns ?? args.config?.repairTurns ?? 12,
  });
}

export default {
  runCoder,
  runRepair,
  CODER_TOOLS,
  DEFAULT_ALLOWED_COMMANDS,
  allowedCommandsFor,
  commandRefusal,
  turnDetail,
  NO_PROGRESS_WINDOW,
};
