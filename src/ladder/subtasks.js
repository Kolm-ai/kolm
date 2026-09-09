// kolm code - subtask decomposition helpers (spec 4.4). Owner ORCH.
//
// The orchestrator model is allowed to split one task into up to four subtasks
// that touch disjoint files. This module holds the pure part of that: validating
// the DAG the plan describes, ordering it into levels that can run in parallel,
// deriving candidate ids and per-subtask plans, and building the feedback text a
// merge conflict turns into.
//
// Everything here is offline and side-effect free. The orchestrator owns the
// worktrees, the events and the money; keeping the graph logic separate is what
// makes the failure modes (a cycle, a dangling depends_on) testable without a
// provider, and what lets the single-unit path stay literally unchanged.

import { LadderError } from './errors.js';
import { candidateIdFor } from './store.js';
import { MAX_SUBTASKS } from './planner.js';

/**
 * Check that a plan's `subtasks` array is a runnable DAG.
 *
 * Rejection is not an error the run dies of: the orchestrator emits
 * `subtasks.rejected` and runs the task as a single unit, which is always a
 * correct (if slower) way to satisfy the same plan.
 *
 * @param {object|null} plan
 * @param {object} [opts]
 * @param {number} [opts.maxSubtasks] ceiling, default MAX_SUBTASKS (4)
 * @returns {{ok:boolean, reason:string|null, subtasks:Array<object>}}
 */
export function validateSubtasks(plan, { maxSubtasks = MAX_SUBTASKS } = {}) {
  const subtasks = Array.isArray(plan?.subtasks) ? plan.subtasks : [];
  if (subtasks.length < 2) return { ok: false, reason: 'no-split', subtasks: [] };
  if (subtasks.length > maxSubtasks) {
    return { ok: false, reason: `too-many-subtasks (${subtasks.length} > ${maxSubtasks})`, subtasks };
  }

  const ids = new Set();
  for (const st of subtasks) {
    const id = typeof st?.id === 'string' ? st.id.trim() : '';
    if (id === '') return { ok: false, reason: 'subtask without an id', subtasks };
    if (ids.has(id)) return { ok: false, reason: `duplicate subtask id ${id}`, subtasks };
    ids.add(id);
  }
  for (const st of subtasks) {
    for (const dep of Array.isArray(st.depends_on) ? st.depends_on : []) {
      if (dep === st.id) return { ok: false, reason: `subtask ${st.id} depends on itself`, subtasks };
      if (!ids.has(dep)) return { ok: false, reason: `unknown depends_on ${dep} in subtask ${st.id}`, subtasks };
    }
  }
  try {
    topoLevels(subtasks);
  } catch (err) {
    return { ok: false, reason: err?.message ?? 'cycle in subtasks', subtasks };
  }
  return { ok: true, reason: null, subtasks };
}

/**
 * Order subtasks into dependency levels. Everything in one level has all its
 * dependencies satisfied by earlier levels, so a level can run in parallel.
 *
 * Within a level the original plan order is preserved, which keeps candidate ids
 * and the merge order deterministic for a given plan.
 *
 * @param {Array<{id:string, depends_on?:string[]}>} subtasks
 * @returns {Array<Array<object>>} levels, each an array of subtask objects
 * @throws {LadderError} `bad-plan` when the graph has a cycle or a dangling edge
 */
export function topoLevels(subtasks) {
  const list = Array.isArray(subtasks) ? subtasks : [];
  const byId = new Map();
  for (const st of list) {
    const id = typeof st?.id === 'string' ? st.id.trim() : '';
    if (id === '') throw new LadderError('bad-plan', 'subtask without an id');
    if (byId.has(id)) throw new LadderError('bad-plan', `duplicate subtask id ${id}`);
    byId.set(id, st);
  }
  for (const st of list) {
    for (const dep of Array.isArray(st.depends_on) ? st.depends_on : []) {
      if (!byId.has(dep)) throw new LadderError('bad-plan', `unknown depends_on ${dep} in subtask ${st.id}`);
    }
  }

  const done = new Set();
  const levels = [];
  let remaining = [...list];
  while (remaining.length > 0) {
    const ready = remaining.filter((st) => (Array.isArray(st.depends_on) ? st.depends_on : [])
      .every((dep) => done.has(dep)));
    if (ready.length === 0) {
      const stuck = remaining.map((st) => st.id).join(', ');
      throw new LadderError('bad-plan', `cycle in subtasks: ${stuck}`, { subtasks: remaining.map((s) => s.id) });
    }
    levels.push(ready);
    for (const st of ready) done.add(st.id);
    const readySet = new Set(ready);
    remaining = remaining.filter((st) => !readySet.has(st));
  }
  return levels;
}

/**
 * Candidate id for one attempt inside one subtask: `s<k><candidateId>`, where
 * `<candidateId>` is exactly today's `c<idx>a<attempt>` (idx = rung * candidates
 * + slot). Prefixing rather than replacing keeps the rung recoverable from the
 * id, keeps ids unique when a rung draws several candidates, and keeps the
 * single-unit ids (no prefix) byte-identical to today.
 *
 * @param {number} k zero-based subtask index, in plan order
 * @param {number} idx candidate index within the ladder
 * @param {number} [attempt] repair round
 * @returns {string} e.g. `s0c0a0`, `s1c3a1`
 */
export function subtaskCandidateIdFor(k, idx, attempt = 0) {
  return `s${Math.max(0, Math.trunc(Number(k)) || 0)}${candidateIdFor(idx, attempt)}`;
}

/**
 * Parse a subtask candidate id back into its parts. Returns null for a plain
 * single-unit id, which is how callers tell the two apart.
 *
 * @param {string} candidateId
 * @returns {{k:number, idx:number, attempt:number}|null}
 */
export function parseSubtaskCandidateId(candidateId) {
  const m = /^s(\d+)c(\d+)a(\d+)$/.exec(String(candidateId ?? ''));
  if (!m) return null;
  return { k: Number(m[1]), idx: Number(m[2]), attempt: Number(m[3]) };
}

// ------------------------------------------------------ scoped acceptance

/** A file stem shorter than this says nothing about ownership ("a.js"). */
const STEM_MIN_CHARS = 3;

/**
 * Stems that name nothing in particular. "test/index.js" and "src/index.js"
 * share the stem "index"; matching it would attribute every criterion that
 * says "index" to whichever subtask happens to own such a file.
 */
const GENERIC_STEMS = new Set([
  'index', 'main', 'mod', 'app', 'lib', 'src', 'test', 'tests', 'spec', 'specs',
  'util', 'utils', 'helper', 'helpers', 'types', 'config', 'readme', 'setup',
]);

/**
 * The name tokens one criterion can mention for a list of files: the path,
 * its basename, and (when specific enough) its stem. Lower-cased, "/"-normal.
 * @param {string[]} files
 * @returns {Set<string>}
 */
function fileTokens(files) {
  const tokens = new Set();
  for (const f of files) {
    const norm = String(f ?? '').replace(/\\/g, '/').trim().toLowerCase();
    if (norm === '') continue;
    tokens.add(norm);
    const base = norm.split('/').pop();
    if (!base) continue;
    tokens.add(base);
    const stem = base.split('.')[0];
    if (stem.length >= STEM_MIN_CHARS && !GENERIC_STEMS.has(stem)) tokens.add(stem);
  }
  return tokens;
}

function escapeRegExp(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Where the criterion text first names any of these tokens, or -1. Paths and
 * basenames match as substrings; bare stems only as whole words, so "toCsv"
 * does not count as a mention of "csv".
 * @param {string} criterion
 * @param {Set<string>} tokens
 * @returns {number}
 */
function firstMention(criterion, tokens) {
  const text = String(criterion ?? '').replace(/\\/g, '/').toLowerCase();
  let best = -1;
  for (const t of tokens) {
    let at = -1;
    if (t.includes('/') || t.includes('.')) {
      at = text.indexOf(t);
    } else {
      const m = new RegExp(`(^|[^a-z0-9_])${escapeRegExp(t)}(?=[^a-z0-9_]|$)`).exec(text);
      if (m) at = m.index + m[1].length;
    }
    if (at >= 0 && (best < 0 || at < best)) best = at;
  }
  return best;
}

/**
 * The sibling files a criterion names (display form, "/"-normal, deduplicated),
 * skipping tokens this subtask shares with them.
 * @param {string} criterion
 * @param {string[]} siblingFiles
 * @param {Set<string>} mine
 * @returns {string[]}
 */
function foreignFilesNamed(criterion, siblingFiles, mine) {
  const named = [];
  for (const f of siblingFiles) {
    const tokens = fileTokens([f]);
    for (const t of mine) tokens.delete(t);
    if (tokens.size === 0 || firstMention(criterion, tokens) < 0) continue;
    const display = String(f).replace(/\\/g, '/').trim();
    if (!named.includes(display)) named.push(display);
  }
  return named;
}

function cleanStrings(value) {
  return (Array.isArray(value) ? value : [])
    .filter((s) => typeof s === 'string' && s.trim() !== '')
    .map((s) => s.trim());
}

/**
 * The acceptance criteria one subtask is judged against.
 *
 * A subtask that carries its own list uses it. Otherwise the parent's criteria
 * are filtered: a criterion that names this subtask's files stays, one that
 * names only other subtasks' files goes, and one that names no subtask file at
 * all (a generic "npm test passes") stays. A criterion that names both sides
 * belongs to whichever file it names first (its subject): "src/csv.js does not
 * import src/table.js" is the csv subtask's, while "test/integration.test.js
 * requires ../src/csv.js" is the test subtask's even though it says csv.js. A
 * kept criterion that still names a foreign file gets a note saying which part
 * is this subtask's, since a grader otherwise fails it for the other half.
 * Judging a subtask against the whole parent list made every reviewer mark
 * other subtasks' criteria unmet, which forced repairs that then widened the
 * diff into the other subtasks' files.
 *
 * When nothing survives, one criterion is synthesised from the subtask itself
 * so the graders still attest to something concrete.
 *
 * @param {object} plan the whole plan (its subtasks tell which files are foreign)
 * @param {object} subtask one entry of plan.subtasks
 * @returns {string[]} never empty when the subtask names at least a title or a file
 */
export function scopedAcceptance(plan, subtask) {
  const own = cleanStrings(subtask?.acceptance);
  if (own.length > 0) return own;

  const parent = cleanStrings(plan?.acceptance);
  const files = Array.isArray(subtask?.files) ? subtask.files : [];
  const mine = fileTokens(files);
  const siblings = (Array.isArray(plan?.subtasks) ? plan.subtasks : []).filter(
    (s) => s !== subtask && (s?.id == null || subtask?.id == null || s.id !== subtask.id),
  );
  const siblingFiles = siblings.flatMap((s) => (Array.isArray(s?.files) ? s.files : []));
  const theirs = fileTokens(siblingFiles);
  // A token both sides own (a shared basename or stem) says nothing about
  // ownership: drop it from both sides so it neither keeps nor drops a criterion.
  for (const t of [...mine]) {
    if (theirs.has(t)) {
      mine.delete(t);
      theirs.delete(t);
    }
  }

  const ownFiles = files.map((f) => String(f).replace(/\\/g, '/').trim()).filter((f) => f !== '');
  const kept = [];
  for (const c of parent) {
    const mineAt = firstMention(c, mine);
    const theirsAt = firstMention(c, theirs);
    if (theirsAt < 0) {
      kept.push(c); // names this subtask only, or no subtask file at all
      continue;
    }
    if (mineAt < 0 || mineAt > theirsAt) continue; // another subtask's criterion
    const foreign = foreignFilesNamed(c, siblingFiles, mine);
    kept.push(
      foreign.length > 0 && ownFiles.length > 0
        ? `${c} (only the ${ownFiles.join(', ')} part is this subtask's; ${foreign.join(', ')} belongs to another subtask)`
        : c,
    );
  }
  if (kept.length > 0) return kept;

  const title = String(subtask?.title ?? subtask?.summary ?? subtask?.id ?? '').trim();
  const where = files.length > 0 ? ` in ${files.join(', ')}` : '';
  const what = title !== '' ? `"${title}"` : 'This subtask';
  return [`${what} is implemented${where} and the existing tests still pass.`];
}

/**
 * The plan one subtask is written and judged against: the whole plan narrowed to
 * this subtask's files and acceptance criteria.
 *
 * `forbidden_paths` is inherited unchanged - a subtask must not touch the
 * lockfiles either - and `files_expected` becomes the subtask's own files so L0
 * scores scope discipline against what this unit actually owns.
 *
 * @param {object} plan the whole plan
 * @param {object} subtask one entry of plan.subtasks
 * @param {number} k its index, for the title
 * @returns {object} a plan-shaped object with no `subtasks` of its own
 */
export function subPlanFor(plan, subtask, k = 0) {
  const files = Array.isArray(subtask?.files) ? subtask.files : [];
  // The planner normalises a missing subtask summary to '' rather than to
  // undefined, so ?? would pick the empty string and the coder would open on a
  // headline-less prompt. Take the first field that actually says something.
  const headline = [subtask?.summary, subtask?.title, subtask?.id]
    .map((v) => String(v ?? '').trim())
    .find((v) => v !== '') ?? '';
  const acceptance = scopedAcceptance(plan, subtask);
  return {
    ...plan,
    title: `${plan?.title ?? 'task'} - subtask ${k + 1}: ${subtask?.title ?? subtask?.id ?? ''}`.slice(0, 200),
    summary: [
      headline,
      '',
      'This is one part of a larger change. Stay inside the files listed for this part;',
      'another coder owns the rest.',
      '',
      `WHOLE TASK: ${String(plan?.summary ?? '').slice(0, 1500)}`,
    ].join('\n').slice(0, 4000),
    acceptance,
    files_expected: files,
    // A subtask is one unit of work; it never carries a split of its own.
    subtasks: [],
    integration_test: null,
    subtask_id: subtask?.id ?? null,
  };
}

/**
 * Feedback text for the coder session that resolves a merge conflict in the
 * integration worktree.
 *
 * @param {object} args
 * @param {string[]} args.conflicts conflicting paths
 * @param {string} [args.subtaskId] the subtask whose merge conflicted
 * @param {string} [args.detail] extra text from the driver (git output, markers)
 * @returns {string} ASCII feedback
 */
export function conflictFeedback({ conflicts = [], subtaskId = null, detail = '' } = {}) {
  const lines = [];
  lines.push('MERGE CONFLICT');
  lines.push(subtaskId
    ? `Merging subtask ${subtaskId} into the integration branch left conflicts.`
    : 'Merging into the integration branch left conflicts.');
  lines.push('Resolve every conflict so the merged tree keeps BOTH changes.');
  lines.push('Remove every conflict marker (<<<<<<<, =======, >>>>>>>) you resolve.');
  if (conflicts.length > 0) {
    lines.push('');
    lines.push('CONFLICTING FILES');
    for (const p of conflicts.slice(0, 40)) lines.push(`  - ${p}`);
  }
  const extra = String(detail ?? '').trim();
  if (extra !== '') {
    lines.push('');
    lines.push('DETAIL');
    lines.push(extra.slice(0, 4000));
  }
  return lines.join('\n');
}

/**
 * Which subtasks a resumed run may reuse instead of running again, read back
 * from the events table (the source of truth for resume).
 *
 * Only a PASS that left a branch behind is reusable. The branch is the whole
 * point - it is what the integration merge consumes - and a FAIL or a JUDGE is
 * exactly the work `resume --coder <stronger>` exists to redo. Found live in
 * r_20260909_f02fbbb9: both subtasks failed, and because every `subtask.end`
 * counted as finished, a resume would have called the run done and rebuilt the
 * same empty integration.
 *
 * The last entry for an id wins, so a retry replaces the verdict before it in
 * both directions: a retry that passes becomes reusable, one that fails again
 * stops being so.
 *
 * @param {Array<{type:string, payload:object}>} events events for the run, in order
 * @returns {Map<string, {id:string, verdict:string, winner:string|null, sha:string|null,
 *   branch:string|null, costUsd:number}>} keyed by subtask id
 */
export function finishedSubtasks(events) {
  const out = new Map();
  for (const ev of Array.isArray(events) ? events : []) {
    if (ev?.type !== 'subtask.end') continue;
    const p = ev.payload ?? {};
    if (typeof p.id !== 'string' || p.id === '') continue;
    const branch = typeof p.branch === 'string' && p.branch !== '' ? p.branch : null;
    if (p.verdict !== 'PASS' || branch === null) {
      out.delete(p.id);
      continue;
    }
    out.set(p.id, {
      id: p.id,
      verdict: 'PASS',
      winner: p.winner ?? null,
      sha: p.sha ?? null,
      branch,
      costUsd: Number(p.costUsd) || 0,
    });
  }
  return out;
}

/**
 * The verdict each subtask last ended with, reusable or not. The orchestrator
 * reads it to say what a retry is retrying (`subtask.retry.priorVerdict`); a
 * subtask that never ended has no entry.
 *
 * @param {Array<{type:string, payload:object}>} events events for the run, in order
 * @returns {Map<string, string>} subtask id -> verdict ('PASS' | 'FAIL' | 'JUDGE')
 */
export function priorSubtaskVerdicts(events) {
  const out = new Map();
  for (const ev of Array.isArray(events) ? events : []) {
    if (ev?.type !== 'subtask.end') continue;
    const p = ev.payload ?? {};
    if (typeof p.id !== 'string' || p.id === '') continue;
    out.set(p.id, typeof p.verdict === 'string' && p.verdict !== '' ? p.verdict : 'FAIL');
  }
  return out;
}

export default {
  validateSubtasks,
  topoLevels,
  subtaskCandidateIdFor,
  parseSubtaskCandidateId,
  subPlanFor,
  conflictFeedback,
  finishedSubtasks,
  priorSubtaskVerdicts,
};
