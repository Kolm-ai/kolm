// kolm code — the planner (spec 6.1). Owner D.
//
// One orchestrator call, before any money is spent on coders, that turns a
// sentence of English into the contract every later rung is measured against:
// acceptance criteria (L3/L4 score against them), expected files and forbidden
// paths (L0 bands scope discipline against them), and the test command (L2 runs
// it). Getting this wrong is cheap here and expensive three rungs later, so the
// planner gets `effort:'medium'` and a strict schema rather than free text.

import { LadderError } from './errors.js';
import { modelTimeoutMs } from './config.js';
import { briefToPrompt } from './context/repo-brief.js';

// Plan v2 (spec 4.1): the brief is now the orchestrator's main lever for
// splitting work, so it gets a much larger slice of the prompt than v1's 10 KB.
const BRIEF_CAP = 32 * 1024;

/** Hard ceiling on subtasks. Beyond this the coordination costs more than it saves. */
export const MAX_SUBTASKS = 4;

const SYSTEM = [
  'You plan one focused code change inside an existing repository.',
  'You do not write the code; you write the contract the change will be judged against.',
  'Acceptance criteria must be checkable by reading the diff or running the test command — no vague intentions.',
  'Name only files the task actually needs. Forbidden paths are globs the change must not touch',
  '(lockfiles, generated output, unrelated packages, CI credentials).',
  'If the task is ambiguous, still produce a plan for the most reasonable reading and list the ambiguities in "clarifications".',
  'Estimate "complexity": trivial (a line or two, one file), small (one file, a contained change),',
  'medium (a few files that must agree), large (a cross-cutting change).',
  'Split the task into "subtasks" ONLY when the parts touch disjoint sets of files and can be written',
  `independently; never more than ${MAX_SUBTASKS}, and leave the array empty when a single change is simpler.`,
  'Each subtask lists the files it owns and may name earlier subtask ids in "depends_on".',
  'Give each subtask its own "acceptance" list: only criteria its files alone can satisfy, never a criterion about another subtask.',
  'When you split, set "integration_test" to the command that proves the merged whole works.',
].join(' ');

/** JSON schema the planner must fill. */
export const PLAN_SCHEMA = Object.freeze({
  name: 'kolm_plan',
  schema: {
    type: 'object',
    required: ['title', 'summary', 'acceptance', 'files_expected', 'forbidden_paths', 'risk'],
    properties: {
      title: { type: 'string' },
      summary: { type: 'string' },
      acceptance: { type: 'array', items: { type: 'string' } },
      files_expected: { type: 'array', items: { type: 'string' } },
      forbidden_paths: { type: 'array', items: { type: 'string' } },
      test_command: { type: 'string' },
      risk: { type: 'string', enum: ['low', 'medium', 'high'] },
      budget_hint_usd: { type: 'number' },
      clarifications: { type: 'array', items: { type: 'string' } },
      complexity: { type: 'string', enum: ['trivial', 'small', 'medium', 'large'] },
      subtasks: {
        type: 'array',
        items: {
          type: 'object',
          required: ['id', 'title', 'files'],
          properties: {
            id: { type: 'string' },
            title: { type: 'string' },
            summary: { type: 'string' },
            files: { type: 'array', items: { type: 'string' } },
            acceptance: { type: 'array', items: { type: 'string' } },
            depends_on: { type: 'array', items: { type: 'string' } },
          },
          additionalProperties: false,
        },
      },
      integration_test: { type: 'string' },
    },
    additionalProperties: false,
  },
});

const DEFAULT_FORBIDDEN = Object.freeze([
  '**/node_modules/**',
  '**/*.lock',
  '**/package-lock.json',
  '**/.git/**',
  '**/.env*',
]);

function strings(value, cap = 40) {
  if (!Array.isArray(value)) return [];
  return value.filter((v) => typeof v === 'string' && v.trim() !== '').map((v) => v.trim()).slice(0, cap);
}

function num(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

/**
 * Normalise the raw `subtasks` array into the shape the orchestrator consumes.
 *
 * Shape only: ids are de-duplicated and unknown `depends_on` references are kept
 * verbatim so that subtasks.js can REJECT the split (and fall back to a single
 * unit) rather than silently running a plan the model did not write. A model
 * that returns one subtask has not split anything, so that collapses to [].
 *
 * @param {unknown} value raw `json.subtasks`
 * @returns {Array<{id:string,title:string,summary:string,files:string[],acceptance:string[],depends_on:string[]}>}
 */
export function normalizeSubtasks(value) {
  if (!Array.isArray(value)) return [];
  const seen = new Set();
  const out = [];
  for (const raw of value.slice(0, 16)) {
    if (!raw || typeof raw !== 'object') continue;
    const id = String(raw.id ?? '').trim().slice(0, 40);
    if (id === '' || seen.has(id)) continue;
    seen.add(id);
    out.push({
      id,
      title: String(raw.title ?? id).slice(0, 200),
      summary: String(raw.summary ?? '').slice(0, 2000),
      files: strings(raw.files, 60),
      acceptance: strings(raw.acceptance, 20),
      depends_on: strings(raw.depends_on, 8),
    });
  }
  return out.length > 1 ? out : [];
}

function buildPrompt({ task, brief, commands }) {
  const parts = [];
  parts.push('TASK');
  parts.push(String(task?.text ?? '').slice(0, 8000));
  if (task?.title) parts.push(`(caller-supplied title: ${task.title})`);
  if (task?.repoUrl) parts.push(`(repository: ${task.repoUrl})`);
  if (task?.baseRef) parts.push(`(base ref: ${task.baseRef})`);
  parts.push('');
  const briefText = briefToPrompt(brief);
  parts.push(briefText.length > BRIEF_CAP ? `${briefText.slice(0, BRIEF_CAP)}\n[truncated]` : briefText);
  parts.push('');
  if (commands && (commands.test || commands.typecheck || commands.lint)) {
    parts.push('DETECTED COMMANDS (reuse these unless the task needs a different one)');
    if (commands.test) parts.push(`  test: ${commands.test}`);
    if (commands.typecheck) parts.push(`  typecheck: ${commands.typecheck}`);
    if (commands.lint) parts.push(`  lint: ${commands.lint}`);
    parts.push('');
  }
  parts.push('Return JSON matching the schema. Keep the plan to the smallest change that satisfies the task.');
  return parts.join('\n');
}

/**
 * Produce the plan for one task.
 *
 * @param {object} args
 * @param {{text:string, repoUrl?:string, baseRef?:string, title?:string}} args.task
 * @param {object} [args.brief] repoBrief() output
 * @param {object} args.orchestrator provider
 * @param {object} [args.store]
 * @param {string} [args.runId]
 * @param {object} [args.config]
 * @param {object} [args.commands] detectCommands() output
 * @param {AbortSignal} [args.signal]
 * @returns {Promise<{title:string, summary:string, acceptance:string[], files_expected:string[],
 *   forbidden_paths:string[], test_command:string|null, risk:'low'|'medium'|'high',
 *   budget_hint_usd:number, clarifications:string[], complexity:'trivial'|'small'|'medium'|'large',
 *   subtasks:Array<object>, integration_test:string|null, model:string|null, costUsd:number}>}
 */
export async function plan({
  task,
  brief,
  orchestrator,
  store,
  runId,
  config = {},
  commands,
  signal,
} = {}) {
  if (!task || typeof task.text !== 'string' || task.text.trim() === '') {
    throw new LadderError('bad-arg', 'plan() needs a task with non-empty text');
  }
  if (!orchestrator || typeof orchestrator.chat !== 'function') {
    throw new LadderError('bad-arg', 'plan() needs an orchestrator provider');
  }
  if (orchestrator.armed === false) {
    throw new LadderError(
      'not-armed',
      `the orchestrator rung ${orchestrator.id ?? '?'} is not armed; run \`kolm code doctor\``,
      { ref: orchestrator.id ?? null },
    );
  }

  const prompt = buildPrompt({ task, brief, commands });
  let res = null;
  let costUsd = 0;
  let lastError = null;
  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      res = await orchestrator.chat({
        system: SYSTEM,
        messages: [{ role: 'user', content: prompt }],
        jsonSchema: PLAN_SCHEMA,
        maxTokens: Number(config.plannerMaxTokens) > 0 ? Number(config.plannerMaxTokens) : 2000,
        effort: 'medium',
        cacheSystem: true,
        signal,
        timeoutMs: modelTimeoutMs(config),
      });
    } catch (err) {
      lastError = err;
      res = null;
    }
    const attemptCost = Number(res?.costUsd) || 0;
    costUsd += attemptCost;
    if (store && runId) {
      store.addCost({
        runId,
        candidateId: null,
        stage: 'plan',
        provider: res?.provider ?? orchestrator.provider ?? null,
        model: res?.model ?? orchestrator.model ?? null,
        inputTokens: res?.usage?.input ?? 0,
        outputTokens: res?.usage?.output ?? 0,
        cacheRead: res?.usage?.cacheRead ?? 0,
        costUsd: attemptCost,
      });
    }
    if (res && res.json != null) break;
    res = null;
  }

  if (!res) {
    if (lastError && lastError.code === 'not-armed') throw lastError;
    throw new LadderError(
      'plan-failed',
      `the orchestrator returned no usable plan${lastError ? `: ${lastError.message}` : ''}`,
      { runId: runId ?? null },
    );
  }

  const json = res.json;
  const testCommand = typeof json.test_command === 'string' && json.test_command.trim() !== ''
    ? json.test_command.trim()
    : null;
  const forbidden = strings(json.forbidden_paths);
  const built = {
    title: String(json.title ?? task.title ?? '').slice(0, 200) || String(task.text).slice(0, 80),
    summary: String(json.summary ?? '').slice(0, 4000),
    acceptance: strings(json.acceptance),
    files_expected: strings(json.files_expected, 100),
    forbidden_paths: [...new Set([...forbidden, ...DEFAULT_FORBIDDEN])],
    test_command: testCommand,
    risk: ['low', 'medium', 'high'].includes(json.risk) ? json.risk : 'medium',
    budget_hint_usd: num(json.budget_hint_usd, 0),
    clarifications: strings(json.clarifications, 20),
    // Plan v2. `complexity` defaults to 'medium' so a v1 plan (and every plan
    // from a model that ignores the new field) routes to today's ladder exactly.
    complexity: ['trivial', 'small', 'medium', 'large'].includes(json.complexity) ? json.complexity : 'medium',
    subtasks: normalizeSubtasks(json.subtasks),
    integration_test: typeof json.integration_test === 'string' && json.integration_test.trim() !== ''
      ? json.integration_test.trim()
      : null,
    model: orchestrator.id ?? res.model ?? null,
    costUsd,
  };

  if (store && runId) store.emit(runId, 'plan', built);
  return built;
}

export default { plan, PLAN_SCHEMA, normalizeSubtasks, MAX_SUBTASKS };
