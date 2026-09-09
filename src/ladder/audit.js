// kolm code - the plan audit (spec 4.8). Owner ORCH.
//
// A repair round assumes the plan is right and the patch is wrong. That is the
// common case, and it is also how a run burns six coder sessions on a criterion
// no patch can satisfy: in run r_20260909_f02fbbb9 the task claimed two LF-only
// files "keep the CRLF line endings", the planner turned the claim into
// acceptance criterion #2, and every candidate of every round was rejected for
// the same criterion until the subtask failed.
//
// So the coordinator treats its own plan as evidence-dependent. When one
// criterion blocks EVERY candidate before and after a repair round, the
// criterion is the suspect, and one bounded orchestrator call - with the task,
// the plan, the best candidate's diff and deterministic facts about the files
// the criterion names - decides whether it is satisfiable at all:
//
//   keep    - a coder could still do this; nothing changes
//   retire  - the repository already contradicts it and the task never asked to
//             change that ("keep X" where X was never true)
//   rewrite - checkable only in different words; the new wording replaces it
//
// Four rules keep this from becoming a liability of its own:
//   1. It is never fatal. A failed call, bad JSON or an abort means every
//      verdict is `keep` and the round proceeds exactly as it does today.
//   2. At most one audit per unit per run, so a wrong `keep` cannot loop and a
//      wrong `retire` cannot cascade.
//   3. Only the criteria the detector named as stuck can be touched, and the
//      last surviving criterion is never retired: a plan with nothing left to
//      attest to would pass every candidate vacuously.
//   4. It never mutates the plan the store already holds. `retire`/`rewrite`
//      build a NEW plan object for the rest of the unit; the `plan.criterion`
//      events are the record of what changed.
//
// Unlike the coach, this is NOT skipped when the orchestrator and the coder are
// the same model: the question is about the plan, not a second opinion on code.

import { modelTimeoutMs } from './config.js';

/** Largest slice of the best candidate's diff the audit is shown. */
export const AUDIT_DIFF_CAP = 8 * 1024;

/** Hard cap on files whose facts are gathered for one audit. */
export const MAX_FACT_FILES = 8;

/** Per-field character caps, so one long model string cannot fill an event row. */
const REASON_CAP = 300;
const REWRITE_CAP = 300;

const SYSTEM = [
  'You are the orchestrator of a code ladder, auditing your own plan.',
  'Every candidate patch has been rejected for the same acceptance criterion, before and after a repair round.',
  'Decide, for each criterion listed, whether a change to THIS repository as it stands could ever satisfy it.',
  'Answer "retire" when the repository already contradicts the criterion and the task does not ask to change that',
  '(a "keep X" or "preserve X" criterion where X was never true of these files).',
  'Answer "rewrite" when the intent is sound but the wording cannot be checked, and give the new wording.',
  'Answer "keep" for anything a coder could still do; when in doubt, keep.',
  'The file facts below are measured, not reported by a model: trust them over the task text.',
].join(' ');

/** JSON schema the audit must fill. */
export const AUDIT_SCHEMA = Object.freeze({
  name: 'kolm_plan_audit',
  schema: {
    type: 'object',
    required: ['verdicts'],
    properties: {
      verdicts: {
        type: 'array',
        items: {
          type: 'object',
          required: ['index', 'verdict', 'reason'],
          properties: {
            index: { type: 'integer' },
            verdict: { type: 'string', enum: ['keep', 'retire', 'rewrite'] },
            reason: { type: 'string' },
            rewrite: { type: 'string' },
          },
          additionalProperties: false,
        },
      },
    },
    additionalProperties: false,
  },
});

/* ------------------------------------------------------------ file facts */

/**
 * Deterministic facts about one file's content. This is the evidence that
 * settles a claim like "the files use CRLF": nothing here is a model's opinion.
 *
 * `crlf` counts "\r\n" pairs and `lf` counts the newlines that are NOT part of
 * one, so a CRLF file reports `lf: 0` rather than counting every line twice.
 *
 * @param {string|null|undefined} text file content, or null when it is missing
 * @returns {{bytes:number, lines:number, crlf:number, lf:number, bom:boolean,
 *   trailingNewline:boolean}|null} null when there is no content to measure
 */
export function fileFacts(text) {
  if (typeof text !== 'string') return null;
  const bom = text.charCodeAt(0) === 0xfeff;
  const body = bom ? text.slice(1) : text;
  const crlf = (body.match(/\r\n/g) || []).length;
  const newlines = (body.match(/\n/g) || []).length;
  const trailingNewline = body.endsWith('\n');
  return {
    bytes: Buffer.byteLength(text, 'utf8'),
    // A file with no trailing newline still has a last line.
    lines: body === '' ? 0 : newlines + (trailingNewline ? 0 : 1),
    crlf,
    lf: newlines - crlf,
    bom,
    trailingNewline,
  };
}

/**
 * The same facts for a set of paths, in the candidate worktree and in the base
 * commit it started from. A path that is missing, unreadable, or that the
 * driver cannot serve gives `null` for that side rather than failing the audit.
 *
 * @param {object|null} wt candidate worktree (needs `readFile`; `readFileAt` for the base)
 * @param {string[]} paths repository-relative paths
 * @param {object} [opts]
 * @param {string} [opts.baseRef] ref to read the base from; defaults to `wt.baseSha()`
 * @returns {Promise<Array<{path:string, facts:object|null, base:object|null}>>}
 */
export async function fileFactsFor(wt, paths, { baseRef } = {}) {
  const wanted = [];
  for (const raw of Array.isArray(paths) ? paths : []) {
    if (typeof raw !== 'string' || raw.trim() === '') continue;
    const p = raw.trim();
    if (!wanted.includes(p)) wanted.push(p);
    if (wanted.length >= MAX_FACT_FILES) break;
  }
  if (wanted.length === 0 || !wt) return [];

  let ref = typeof baseRef === 'string' && baseRef !== '' ? baseRef : null;
  if (ref === null && typeof wt.baseSha === 'function') {
    try {
      const sha = await wt.baseSha();
      ref = typeof sha === 'string' && sha !== '' ? sha : null;
    } catch {
      ref = null;
    }
  }

  const out = [];
  for (const p of wanted) {
    let facts = null;
    let base = null;
    if (typeof wt.readFile === 'function') {
      try {
        facts = fileFacts(await wt.readFile(p));
      } catch {
        // A path outside the worktree, or a directory: not evidence, not fatal.
      }
    }
    if (ref !== null && typeof wt.readFileAt === 'function') {
      try {
        base = fileFacts(await wt.readFileAt(ref, p));
      } catch {
        // A driver that cannot read history contributes no base facts.
      }
    }
    out.push({ path: p, facts, base });
  }
  return out;
}

/** One measured file rendered as a prompt line. */
function factLine(path, facts, suffix = '') {
  return [
    `${path}${suffix}: `,
    `${facts.lines} lines, ${facts.crlf} CRLF, ${facts.lf} LF, `,
    facts.bom ? 'BOM, ' : 'no BOM, ',
    facts.trailingNewline ? 'trailing newline' : 'no trailing newline',
  ].join('');
}

/**
 * The FILE FACTS block: one line per file, plus one for the base when it
 * differs from the working tree. Files that exist on neither side are dropped,
 * which is also what restricts the path guesses to paths that really exist.
 *
 * @param {Array<{path:string, facts:object|null, base:object|null}>} entries fileFactsFor() output
 * @returns {string[]} lines, empty when nothing could be measured
 */
export function renderFileFacts(entries) {
  const lines = [];
  for (const entry of Array.isArray(entries) ? entries : []) {
    const { path, facts, base } = entry || {};
    if (typeof path !== 'string' || path === '') continue;
    if (facts) lines.push(factLine(path, facts));
    else if (base) lines.push(`${path}: not present in the working tree`);
    else continue;
    if (base && facts && (base.crlf !== facts.crlf || base.lf !== facts.lf)) {
      lines.push(factLine(path, base, ' (base)'));
    }
  }
  return lines;
}

/**
 * Paths a criterion names, by shape alone: a dotted token that looks like a
 * file. Deliberately conservative - the caller measures every candidate and
 * keeps only the ones the worktree actually has, so a wrong guess costs a read.
 *
 * @param {string[]} texts criterion texts
 * @returns {string[]} "/"-normal, deduplicated, capped
 */
export function pathsNamedIn(texts) {
  const out = [];
  for (const raw of Array.isArray(texts) ? texts : []) {
    const text = String(raw ?? '').replace(/\\/g, '/');
    for (const m of text.matchAll(/[\w./-]+\.[a-z]{1,5}\b/g)) {
      const p = m[0];
      if (p.startsWith('/') || p.includes('..')) continue;
      if (!out.includes(p)) out.push(p);
      if (out.length >= MAX_FACT_FILES) return out;
    }
  }
  return out;
}

/* --------------------------------------------------------- reason parsing */

// L3 writes `acceptance #<k> not met: <text>` (l3-semantic.js), where <text>
// may carry the scoped-acceptance annotation in parentheses. L4's rationale
// leads with `Unmet acceptance: #<k> <text>; #<k> <text>` (unmetPrefix), which
// the orchestrator copies into the repair feedback. Both number criteria from
// 1; the indices returned here are 0-based positions in `plan.acceptance`,
// which is the numbering every event and every helper in this file uses.
const NOT_MET_RE = /acceptance\s*#\s*(\d+)\s+not met/gi;
const UNMET_PREFIX_RE = /unmet acceptance\s*:/i;
const HASH_RE = /#\s*(\d+)/g;

/**
 * The acceptance criteria one reason names, 0-based.
 *
 * @param {string} reason one entry of a candidate row's `reasons`
 * @returns {number[]} ascending, deduplicated; empty for a reason that names none
 *   (`judge blocked: spec_compliance`, `rubric_score 0.400 is bad`, ...)
 */
export function acceptanceIndicesIn(reason) {
  const text = String(reason ?? '');
  const found = new Set();
  const add = (raw) => {
    const n = Number(raw);
    if (Number.isInteger(n) && n >= 1) found.add(n - 1);
  };
  for (const m of text.matchAll(NOT_MET_RE)) add(m[1]);
  const at = text.search(UNMET_PREFIX_RE);
  if (at >= 0) {
    for (const m of text.slice(at).matchAll(HASH_RE)) add(m[1]);
  }
  return [...found].sort((a, b) => a - b);
}

/**
 * Every acceptance index a row's reasons name, 0-based.
 * @param {string[]} reasons
 * @returns {number[]} ascending, deduplicated
 */
export function acceptanceIndicesOf(reasons) {
  const found = new Set();
  for (const reason of Array.isArray(reasons) ? reasons : []) {
    for (const k of acceptanceIndicesIn(reason)) found.add(k);
  }
  return [...found].sort((a, b) => a - b);
}

/**
 * A candidate's failure, as a string that is equal for two rounds that failed
 * the same way: the rejecting stage, the acceptance indices, and the reasons
 * that name no criterion (whitespace-collapsed, so only real differences count).
 *
 * @param {{stage?:string, reasons?:string[]}} row
 * @returns {string}
 */
export function reasonSignature(row) {
  const reasons = Array.isArray(row?.reasons) ? row.reasons : [];
  const other = [];
  for (const reason of reasons) {
    if (acceptanceIndicesIn(reason).length > 0) continue;
    const s = String(reason ?? '').trim().replace(/\s+/g, ' ');
    if (s !== '' && !other.includes(s)) other.push(s);
  }
  other.sort();
  return [
    `stage=${row?.stage ?? '?'}`,
    `acceptance=${acceptanceIndicesOf(reasons).join(',')}`,
    `other=${other.join(' | ')}`,
  ].join(';');
}

/**
 * Was this repair round stuck, and on what?
 *
 * A criterion is STUCK when every failing candidate of this round names it AND
 * every failing candidate of the round the repair started from named it - one
 * candidate that got past it means the criterion is satisfiable and the patches
 * were the problem. A candidate made NO PROGRESS when its failure signature is
 * the one it already had, which is the evidence that another identical repair
 * round would buy nothing. A PASS anywhere in either round means nothing was
 * stuck at all.
 *
 * @param {object} args
 * @param {Array<{candidateId:string, verdict:string, stage?:string, reasons?:string[]}>} args.before
 *   rows the repair started from
 * @param {Array<{candidateId:string, sourceId?:string|null, verdict:string, stage?:string,
 *   reasons?:string[]}>} args.after rows the repair produced
 * @returns {{criteria:number[], candidates:string[]}} 0-based criterion indices;
 *   candidate ids are this round's (the ids a next round would repair again)
 */
export function stuckAnalysis({ before = [], after = [] } = {}) {
  const empty = { criteria: [], candidates: [] };
  const beforeRows = Array.isArray(before) ? before : [];
  const afterRows = Array.isArray(after) ? after : [];
  if ([...beforeRows, ...afterRows].some((row) => row?.verdict === 'PASS')) return empty;

  const failingBefore = beforeRows.filter((row) => row?.verdict !== 'PASS');
  const failingAfter = afterRows.filter((row) => row?.verdict !== 'PASS');
  if (failingBefore.length === 0 || failingAfter.length === 0) return empty;

  let criteria = null;
  for (const row of [...failingBefore, ...failingAfter]) {
    const named = new Set(acceptanceIndicesOf(row?.reasons));
    if (named.size === 0) {
      criteria = new Set();
      break;
    }
    criteria = criteria === null ? named : new Set([...criteria].filter((k) => named.has(k)));
  }

  const priorSignature = new Map(beforeRows.map((row) => [row.candidateId, reasonSignature(row)]));
  const candidates = [];
  for (const row of failingAfter) {
    const prior = row?.sourceId == null ? undefined : priorSignature.get(row.sourceId);
    if (prior !== undefined && prior === reasonSignature(row)) candidates.push(String(row.candidateId));
  }

  return {
    criteria: [...(criteria ?? [])].sort((a, b) => a - b),
    candidates: candidates.sort(),
  };
}

/* ---------------------------------------------------------- plan surgery */

/**
 * The plan's acceptance criteria with the numbering the graders use.
 *
 * `index` is the 0-based position in the list L3/L4 are shown (criterion
 * `acceptance #k` is `index k - 1`), and `rawIndex` is where it sits in
 * `plan.acceptance`. The two differ when the plan carries a blank or non-string
 * entry, which `acceptanceList()` in l3-semantic.js drops - this must keep
 * dropping exactly the same entries.
 *
 * @param {object} [plan]
 * @returns {Array<{index:number, rawIndex:number, text:string}>}
 */
export function criterionEntries(plan) {
  const raw = Array.isArray(plan?.acceptance) ? plan.acceptance : [];
  const out = [];
  raw.forEach((text, rawIndex) => {
    if (typeof text !== 'string' || text.trim() === '') return;
    out.push({ index: out.length, rawIndex, text });
  });
  return out;
}

/**
 * Normalise the model's verdicts: known indices only, known verdicts only, and
 * only for the criteria the detector actually asked about.
 *
 * @param {unknown} value raw `json.verdicts`
 * @param {object} [opts]
 * @param {Iterable<number>} [opts.allowed] indices the audit was asked about
 * @returns {Array<{index:number, verdict:'keep'|'retire'|'rewrite', reason:string, rewrite:string|null}>}
 */
export function normalizeVerdicts(value, { allowed } = {}) {
  const permitted = allowed ? new Set(allowed) : null;
  const seen = new Set();
  const out = [];
  for (const raw of Array.isArray(value) ? value : []) {
    if (!raw || typeof raw !== 'object') continue;
    const index = Number(raw.index);
    if (!Number.isInteger(index) || index < 0) continue;
    if (permitted && !permitted.has(index)) continue;
    if (seen.has(index)) continue;
    const verdict = String(raw.verdict ?? '').trim().toLowerCase();
    if (verdict !== 'keep' && verdict !== 'retire' && verdict !== 'rewrite') continue;
    seen.add(index);
    const rewrite = typeof raw.rewrite === 'string' ? raw.rewrite.trim().replace(/\s+/g, ' ').slice(0, REWRITE_CAP) : '';
    out.push({
      index,
      verdict,
      reason: String(raw.reason ?? '').trim().replace(/\s+/g, ' ').slice(0, REASON_CAP),
      rewrite: rewrite === '' ? null : rewrite,
    });
  }
  out.sort((a, b) => a.index - b.index);
  return out;
}

/**
 * A NEW plan with the audited criteria removed or reworded. The plan handed in
 * is never mutated: it is the object the store already has on record.
 *
 * A `rewrite` with no new wording, and a `retire` that would leave the plan
 * with nothing to attest to, both degrade to `keep`.
 *
 * @param {object} plan
 * @param {Array<{index:number, verdict:string, reason:string, rewrite:string|null}>} verdicts
 * @returns {{plan:object, changed:boolean, changes:Array<{index:number, verdict:string,
 *   reason:string, from:string, to:string|null}>}}
 */
export function amendPlan(plan, verdicts) {
  const entries = criterionEntries(plan);
  const byIndex = new Map(entries.map((e) => [e.index, e]));
  const drop = new Set();
  const replace = new Map();
  const changes = [];
  for (const v of Array.isArray(verdicts) ? verdicts : []) {
    const entry = byIndex.get(v?.index);
    if (!entry) continue;
    if (v.verdict === 'retire') {
      // Never leave the plan with nothing to attest to: a criterion-free plan
      // passes every candidate on the rubric alone.
      if (entries.length - drop.size <= 1) continue;
      drop.add(entry.rawIndex);
      changes.push({ index: entry.index, verdict: 'retire', reason: v.reason ?? '', from: entry.text, to: null });
    } else if (v.verdict === 'rewrite') {
      const to = typeof v.rewrite === 'string' ? v.rewrite : '';
      if (to === '' || to === entry.text) continue;
      replace.set(entry.rawIndex, to);
      changes.push({ index: entry.index, verdict: 'rewrite', reason: v.reason ?? '', from: entry.text, to });
    }
  }
  if (changes.length === 0) return { plan, changed: false, changes: [] };

  const acceptance = [];
  (Array.isArray(plan?.acceptance) ? plan.acceptance : []).forEach((text, i) => {
    if (drop.has(i)) return;
    acceptance.push(replace.has(i) ? replace.get(i) : text);
  });
  return { plan: { ...plan, acceptance }, changed: true, changes };
}

/**
 * Re-apply the `plan.criterion` events of an earlier attempt at this run, so a
 * resumed run judges against the plan the audit left behind instead of the one
 * the planner wrote. Criteria are matched by their text, which is what the
 * events record, so an amendment applies wherever the criterion ended up.
 *
 * @param {object} plan the unit's plan as the planner wrote it
 * @param {Array<{type:string, payload:object}>} events stored run events
 * @param {number|null} unit subtask index, or null for a single-unit run
 * @returns {{plan:object, changed:boolean, applied:number}}
 */
export function applyPlanCriterionEvents(plan, events, unit = null) {
  let current = plan;
  let applied = 0;
  for (const event of Array.isArray(events) ? events : []) {
    if (event?.type !== 'plan.criterion') continue;
    const payload = event.payload && typeof event.payload === 'object' ? event.payload : {};
    const at = payload.unit === undefined ? null : payload.unit;
    if ((at ?? null) !== (unit ?? null)) continue;
    const entry = criterionEntries(current).find((e) => e.text === payload.from);
    if (!entry) continue;
    const out = amendPlan(current, [{
      index: entry.index,
      verdict: String(payload.verdict ?? ''),
      reason: String(payload.reason ?? ''),
      rewrite: typeof payload.to === 'string' ? payload.to : null,
    }]);
    if (!out.changed) continue;
    current = out.plan;
    applied += 1;
  }
  return { plan: current, changed: applied > 0, applied };
}

/* ------------------------------------------------------------- the audit */

/** Render the audit prompt. ASCII only; no key, no path outside the repo. */
function buildPrompt({ task, plan, criteria, diffText, factLines }) {
  const parts = [];
  parts.push('TASK AS THE OPERATOR WROTE IT');
  parts.push(String(task ?? '(none)').slice(0, 4000));
  parts.push('');
  parts.push('PLAN');
  parts.push(String(plan?.title ?? '(untitled)'));
  if (plan?.summary) parts.push(String(plan.summary).slice(0, 2000));
  parts.push('');
  parts.push('CRITERIA UNDER AUDIT (every candidate has failed these, before and after a repair)');
  for (const c of criteria) parts.push(`  index ${c.index}: ${String(c.text).slice(0, 500)}`);
  parts.push('');
  const diff = String(diffText ?? '');
  parts.push('BEST CANDIDATE DIFF');
  parts.push(diff.trim() === ''
    ? '(empty diff - the candidate produced no change)'
    : (diff.length > AUDIT_DIFF_CAP ? `${diff.slice(0, AUDIT_DIFF_CAP)}\n[truncated]` : diff));
  parts.push('');
  parts.push('FILE FACTS (measured from the repository, not from the task text)');
  if (factLines.length === 0) parts.push('  (no file named by these criteria could be measured)');
  else for (const line of factLines) parts.push(`  ${line}`);
  parts.push('');
  parts.push('Return one verdict per criterion index above, using that same index.');
  parts.push('Return JSON matching the schema.');
  return parts.join('\n');
}

/**
 * Audit the criteria a repair round got stuck on. Never throws.
 *
 * Emits `plan.criterion {unit, round, index, verdict, reason, from, to, costUsd}`
 * once per criterion the audit changed, and nothing at all when every verdict is
 * `keep` - an audit that confirms the plan is not news.
 *
 * @param {object} args
 * @param {object} args.orchestrator orchestrator provider
 * @param {object} args.plan the unit's current plan
 * @param {Array<{index:number, text:string}>} args.criteria stuck criteria, 0-based indices
 * @param {string} [args.task] the task text as the operator wrote it
 * @param {string} [args.diffText] the best failing candidate's diff
 * @param {Array<{path:string, facts:object|null, base:object|null}>|string[]} [args.fileFacts]
 *   fileFactsFor() output, or lines already rendered
 * @param {object} [args.config] reads `audit`, `modelTimeoutMs`
 * @param {object} [args.store]
 * @param {string} [args.runId]
 * @param {number|null} [args.unit] subtask index, or null for a single-unit run
 * @param {number} [args.round] repair round the audit belongs to
 * @param {boolean} [args.budgetExhausted] true when the run has no money left
 * @param {AbortSignal} [args.signal]
 * @returns {Promise<{plan:object, changed:boolean, changes:Array<object>,
 *   verdicts:Array<object>, costUsd:number, skipped:string|null}>}
 */
export async function auditCriteria({
  orchestrator,
  plan,
  criteria = [],
  task,
  diffText,
  fileFacts: facts = [],
  config = {},
  store,
  runId,
  unit = null,
  round = 0,
  budgetExhausted = false,
  signal,
} = {}) {
  const unchanged = { plan, changed: false, changes: [], verdicts: [], costUsd: 0 };
  const wanted = (Array.isArray(criteria) ? criteria : [])
    .filter((c) => c && Number.isInteger(Number(c.index)) && typeof c.text === 'string')
    .map((c) => ({ index: Number(c.index), text: c.text }));

  if (config.audit === false) return { ...unchanged, skipped: 'disabled' };
  if (budgetExhausted) return { ...unchanged, skipped: 'budget' };
  if (!orchestrator || typeof orchestrator.chat !== 'function' || orchestrator.armed === false) {
    return { ...unchanged, skipped: 'no-orchestrator' };
  }
  if (wanted.length === 0) return { ...unchanged, skipped: 'no-criteria' };

  const factLines = (Array.isArray(facts) ? facts : []).every((f) => typeof f === 'string')
    ? facts.filter((f) => typeof f === 'string' && f !== '')
    : renderFileFacts(facts);

  let res = null;
  try {
    res = await orchestrator.chat({
      system: SYSTEM,
      messages: [{
        role: 'user',
        content: buildPrompt({ task, plan, criteria: wanted, diffText, factLines }),
      }],
      jsonSchema: AUDIT_SCHEMA,
      maxTokens: 800,
      effort: 'low',
      cacheSystem: true,
      signal,
      timeoutMs: modelTimeoutMs(config),
    });
  } catch {
    // Never fatal: an audit that cannot answer leaves the plan exactly as it is.
    res = null;
  }

  const costUsd = Number(res?.costUsd) || 0;
  if (res && store && runId) {
    try {
      store.addCost({
        runId,
        candidateId: null,
        stage: 'audit',
        provider: res?.provider ?? orchestrator.provider ?? null,
        model: res?.model ?? orchestrator.model ?? null,
        inputTokens: res?.usage?.input ?? 0,
        outputTokens: res?.usage?.output ?? 0,
        cacheRead: res?.usage?.cacheRead ?? 0,
        costUsd,
      });
    } catch { /* the ledger is best-effort here; the run must not die for an audit */ }
  }

  const verdicts = normalizeVerdicts(res?.json?.verdicts, { allowed: wanted.map((c) => c.index) });
  const amended = amendPlan(plan, verdicts);
  if (amended.changed && store && runId) {
    for (const change of amended.changes) {
      try {
        store.emit(runId, 'plan.criterion', {
          unit,
          round,
          index: change.index,
          verdict: change.verdict,
          reason: change.reason,
          from: change.from,
          to: change.to,
          costUsd,
        });
      } catch { /* as above */ }
    }
  }
  return {
    plan: amended.plan,
    changed: amended.changed,
    changes: amended.changes,
    verdicts,
    costUsd,
    skipped: null,
  };
}

export default {
  auditCriteria,
  fileFacts,
  fileFactsFor,
  renderFileFacts,
  pathsNamedIn,
  acceptanceIndicesIn,
  acceptanceIndicesOf,
  reasonSignature,
  stuckAnalysis,
  criterionEntries,
  normalizeVerdicts,
  amendPlan,
  applyPlanCriterionEvents,
  AUDIT_SCHEMA,
};
