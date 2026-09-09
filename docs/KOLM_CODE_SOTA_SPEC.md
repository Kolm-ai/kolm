# kolm code: SOTA coordination wave (2026-09-09)

This spec turns `kolm code` from a single-plan confidence cascade into a
coordination agent: an orchestrator model that decomposes, routes, coaches and
integrates the work of cheaper coder models, with a context pack that finds the
right code before the first turn, a coder runtime that survives long
transcripts and models without native tool calling, and a bundled benchmark
that measures every one of these claims on a local model.

It builds on what already ships (see `docs/KOLM_CODE.md`): ladder of coder
rungs, N candidates per rung in isolated worktrees, stages l0-l4 with
calibrated bands, repair rounds, escalation, resume, budget stop, receipts,
keystore, hardening (commits 8878007e, 387828bc, c53a4893 on `kolm-code-ladder`).

## 0. Invariants (every agent, every file)

- Node 22+, ESM, no new npm dependencies, no network in tests. Tests use
  `createFakeDriver` / `createFakeProvider` (`src/ladder/worktree/fake.js`,
  `src/ladder/providers/fake.js`) and `node:test`.
- `node --test --test-concurrency=1 tests/ladder-*.test.js` stays green;
  `npx eslint src/ladder cli/kolm-code.mjs tests/ladder-*.test.js` stays clean.
- CLI and event strings are ASCII only. The word spelled h-o-n-e-s-t (and
  every derivative) never appears in code, docs, tests or output; write
  "Caveats", "Constraints" or "Limitations" instead. Never print a secret;
  `scrubSecrets` on every model error.
- Every model call goes through `store.addCost` so the budget stop stays exact.
- Resume must still work: a run killed at any point resumes from its stored
  events/candidates. New state that resume needs goes into events (payloads
  are free-form JSON) or into a store migration (bump `schema_meta`).
- Every new behaviour is on by default only when it cannot make a simple task
  worse; otherwise it is opt-in via a `KOLM_CODE_*` variable listed in section 7.
- Own only your files (section 8). If you need a change in another owner's file,
  write the exact patch you need into your final report; do not apply it.
- Windows is a first-class host: paths through `path.posix` for worktree
  files, CRLF-tolerant matching, cmd.exe has no POSIX tools.
- JSDoc on every export; comments say why, not what.

## 1. Context pack (owner CTX)

### 1.1 Task-focused retrieval: `src/ladder/context/retrieve.js`

```js
export async function retrieveContext(wt, task, opts = {}) -> {
  terms: string[],           // identifiers/paths pulled from the task text
  files: [{ path, score, why: string }],   // ranked, at most opts.maxFiles (12)
  snippets: [{ path, startLine, endLine, text, score }], // at most opts.maxBytes (16 KB total)
  bytes: number,
}
export function retrievalToPrompt(retrieval) -> string   // ASCII, bounded
```

- Term extraction: backtick spans, CamelCase/snake_case identifiers, file
  paths, quoted strings, function-call shapes (`foo(`) from `task.text` and
  `task.title`. Drop stop words and terms under 3 chars.
- Candidate files: `wt.glob` over the tree (respect `isVendored` from
  repo-brief; skip binaries, lockfiles, >256 KB files). Score = weighted sum of
  term hits in path (x3), in exports/definitions (x2), in body (x1), plus a
  bonus when the plan/task names the path outright. Use `wt.search` where the
  driver has it (it does: local-git and fake both expose `search(regex)`),
  falling back to reading files.
- Snippets: for each top file, the definition or first-hit region with +-12
  lines of context, merged when overlapping; whole file when under 60 lines.
  Test files that name a top file's basename get included as one snippet each
  (the coder needs to see the existing test style).
- Deterministic ordering (score desc, path asc). No model calls.
- `retrievalToPrompt` renders `RELEVANT CODE (found by searching the task
  terms; read the full files before editing)` with `path:start-end` headers.

### 1.2 Project instructions: `src/ladder/context/instructions.js`

```js
export async function loadInstructions(wt, opts = {}) -> {
  files: [{ path, bytes }], text: string  // ASCII-safe, capped at 6 KB total
}
```

Read, in priority order and only when present: `AGENTS.md`, `CLAUDE.md`,
`.cursorrules`, `.github/copilot-instructions.md`, `CONTRIBUTING.md` (head,
2 KB). Each file contributes at most 3 KB; the first two get priority. Output
is prefixed `PROJECT INSTRUCTIONS (from <paths>; follow them)`.

### 1.3 Repo brief upgrades: `src/ladder/context/repo-brief.js`

- `parseExports` grows definition extraction for Python (`def`, `class`), Go
  (`func`, `type`), Rust (`pub fn`, `pub struct`, `pub enum`, `impl`), Ruby
  (`def`, `class`, `module`), Java/Kotlin/C# (public classes and methods, one
  level). Keep the 20-name cap per file, the 60-file cap and the 12 KB brief.
- `briefToPrompt(brief)` renders three optional fields when they are present
  on the brief object: `brief.instructions` (from section 1.2, after the README head),
  `brief.lessons` (from section 1.5, via `lessonsToPrompt`) and `brief.retrieval`
  (from section 1.1, via `retrievalToPrompt`, rendered LAST so a consumer's byte cap
  cuts snippets before it cuts the tree). Each keeps its own cap. A brief
  without those fields renders byte for byte as today (tests pin this). The
  12 KB `BRIEF_BYTE_CAP` applies to the core brief only; the three extras add
  up to 6 KB + 1.5 KB + `KOLM_CODE_RETRIEVAL_BYTES`.
- `repoBrief` records `languages: {js, ts, py, go, rs, ...counts}` and
  `testDirs: string[]`.

### 1.4 Command detection: `src/ladder/context/detect.js`

Add: Makefile targets `test`/`check`/`lint` (used only when nothing better is
found), `pyproject` with `[tool.ruff]` -> `lint: ruff check .`, `mypy` ->
`typecheck: mypy .`, `deno.json` -> `deno test`, `pnpm-workspace.yaml` /
`package.json` `workspaces` -> `monorepo: true` with the workspace list, and
`parseTestOutput` support for `unittest` and `deno test` output.

### 1.5 Lessons across runs: `src/ladder/lessons.js`

```js
export function lessonsPath(config, repoKey) -> string    // <dataDir>/code/lessons/<repoKey>.json
export function loadLessons(config, repoKey) -> { version: 1, repoKey, lessons: [{ id, text, runId, ts, hits }] }
export function saveLessons(config, repoKey, lessons) -> void   // atomic write, 0600
export function mergeLessons(existing, proposed, opts) -> lessons  // dedupe (case-insensitive prefix), cap 12, newest wins
export function lessonsToPrompt(lessons) -> string   // 'LESSONS FROM EARLIER RUNS IN THIS REPO' block, <= 1.5 KB
export function extractLessonCandidates(events) -> string[]   // heuristic, no model call
```

`extractLessonCandidates` reads a run's events and proposes at most 5 lessons
from hard evidence only: repair reasons that recurred across candidates, a
`run` tool result that was refused or timed out, a test command the plan
overrode, an edit_file miss on a CRLF file, a stage that skipped at baseline.
Each lesson is one ASCII sentence <= 160 chars naming the file or command.
`repoKey` is what `repo-brief.js` already computes (`repoKeyFor`); export it.
`KOLM_CODE_LESSONS=0` disables load and save.

Tests: `tests/ladder-context.test.js` (extend), `tests/ladder-lessons.test.js`
(new). Retrieval must be tested on a fake worktree with 30+ files where the
task names one identifier and the expected file ranks first.

## 2. Coder runtime (owner CODER)

### 2.1 Transcript compaction: `src/ladder/transcript.js`

```js
export function estimateTokens(messages) -> number       // chars/3.6 for text, exact for tool_result strings, +12 per block
export function compactTranscript(messages, opts) -> { messages, dropped: number, savedTokens: number, note: string|null }
```

Policy (`opts`: `budgetTokens`, `keepRecentTurns` default 6, `minToolResultChars` default 400):

1. Never touch the first user message (the opening: plan, brief, retrieval).
2. Walk from the oldest tool turn forward; replace each `tool_result` content
   older than the last `keepRecentTurns` assistant turns with a one-line stub
   `[trimmed <tool> <path-or-cmd>: <n> chars; re-run the tool if you need it]`
   until `estimateTokens` fits under `budgetTokens`. Keep `edit_file` /
   `write_file` results in full (they are short and are the coder's record of
   what changed).
3. If still over budget, collapse the oldest 50% of the turns into one
   assistant summary block `[earlier turns: read X, Y; edited Z; ran T (exit 0)]`
   built mechanically from the tool calls (no model call).
4. Return `note` for the coder to emit as a `coder.compact` event.

`runCoder`/`runRepair` call it before every model call when
`estimateTokens(messages) > budget`, where budget =
`min(catalog ctx for the model, KOLM_CODE_CONTEXT_TOKENS) * 0.75 - maxTokens`.
Use `describeModel(coder.id).ctx` (from `providers/registry.js`, existing
export) with a 32768 fallback for `ollama:*`.

### 2.2 Self-review before finish (`coder.js`)

When the coder calls `finish` and `config.selfReview !== false`, and the
worktree diff is non-empty, and no self-review has happened yet in this
session: do NOT end. Append a user message:

```
SELF-REVIEW before finish. This is your diff:
<diff, capped at 24 KB>
Check every acceptance criterion:
  - ...
Check: no unrelated files, no leftover debug output, tests you added actually run.
If something is wrong, fix it with tools now. If it is correct, call finish again.
```

Continue the loop; a second `finish` ends the session. The review costs one
turn (it counts against the turn cap). Emit `coder.self_review` with
`{ candidateId, turn, diffBytes, changedFiles }`. Skip when
`maxTurns - turn < 2` (no room) and emit the event with `skipped: 'no-turns'`.

### 2.3 Tool-result hygiene (`coder.js`)

- `read_file` accepts optional `start_line`/`end_line`; a whole-file read over
  400 lines returns the first 400 lines plus `[... N more lines; pass
  start_line/end_line]`.
- `search` returns at most 40 hits, and says how many were cut.
- `run` output: keep the head 4 KB and the tail 4 KB when longer, with a
  marker in between; the tail is where test runners put the summary.
- Tool results carry a per-call byte cap of 16 KB.
- `BRIEF_CAP` in `coder.js` rises from 8 KB to 40 KB so the retrieval snippets
  and instructions that repo-brief now renders (section 1.3) reach the coder intact;
  the transcript budget (section 2.1) is what protects small models, not this cap.

### 2.4 Steering hints (`coder.js`)

`runCoder` and `runRepair` accept `hints: string[]` (from the coach, section 4.3)
and render them in the opening under `HINTS FROM THE ORCHESTRATOR` (each <= 240
chars, at most 6).

### 2.5 Events

`coder.turn` payload gains `estTokens` and `compacted: boolean`. New events:
`coder.compact` `{ candidateId, turn, dropped, savedTokens, note }`,
`coder.self_review`,
`coder.protocol` `{ candidateId, turn, protocol: 'text', why: 'no-tool-calls' }`
(once per session, when the loop moves a coder that has never called a tool to
the text tool protocol; see section 4.3).

Tests: `tests/ladder-transcript.test.js` (new), `tests/ladder-hardening.test.js`
must keep passing unchanged; add coder tests to `tests/ladder-coder.test.js`
(new file; do not edit `ladder-hardening.test.js`).

## 3. Providers (owner PROV)

### 3.1 Text tool protocol: `src/ladder/providers/text-tools.js`

For models with no native tool calling (Ollama `deepseek-coder-v2`, many
OpenRouter free models, any endpoint that answers 400 to `tools`).

```js
export function renderToolsForText(tools) -> string        // system-prompt block describing the protocol + each tool's JSON schema
export function renderToolResultsForText(blocks) -> string // user text from tool_result blocks
export function parseTextToolCalls(text) -> { text: string, toolCalls: [{ id, name, input }], errors: string[] }
export const TEXT_TOOL_PROTOCOL_VERSION = 1
```

Protocol: the model emits one or more fenced blocks

```
<<<tool name="edit_file">>>
{"path":"src/a.js","old_string":"...","new_string":"..."}
<<<end>>>
```

Anything outside the fences is assistant text. Parsing is tolerant to code
fences around the block, trailing prose, and CRLF. A block whose JSON does
not parse becomes a `toolCalls` entry with `input: { __parse_error: msg, raw }`
so the coder's executor answers with a readable error and the model can retry
(the coder already answers unknown/invalid input with an error string; verify
it does for this shape and keep the behaviour in your adapter if not).
Ids are `text_<n>` per reply.

### 3.2 Adapter integration (`openai-compatible.js`, `anthropic.js`)

- `describeModel(ref).tools` (new registry field, default `true`; `false` for
  `ollama:deepseek-coder-v2*` and a small known list; `KOLM_CODE_TEXT_TOOLS=
  <comma refs>` forces it) selects the text protocol at construction.
- Runtime fallback: an HTTP 400 whose body matches `/does not support tools/i`
  (Ollama) or `/tool(s|_choice)? (is|are) not supported/i` flips the provider
  instance to the text protocol for the rest of the process and retries the
  same call once. Emit nothing (providers have no store); return
  `meta.textTools: true` on the reply so the coder can note it.
- In text mode: tool definitions go into the system prompt (after the
  caller's system), `tool_result` blocks are rendered as user text, the reply
  text is parsed into `toolCalls` and the returned `content` carries proper
  `tool_use` blocks so the coder's transcript stays uniform.
- `jsonSchema` calls in text mode: instruct JSON-only in the system prompt and
  parse the first balanced JSON object from the reply (the adapter already has
  a JSON extraction path; reuse it).

### 3.3 Rate limiting and backoff: `src/ladder/providers/limiter.js`

```js
export function createLimiter({ concurrency, minIntervalMs }) -> { run(fn, {signal}) , stats() }
export function limiterFor(provider, config) -> limiter   // one per provider name per process
export function retryDelayMs(attempt, headers, opts) -> number  // honours Retry-After (seconds or HTTP date) else exp backoff 1s,2s,4s,8s (+-20% jitter), capped 30s
```

Defaults per provider: anthropic 4, cerebras 2, deepseek 4, openrouter 4,
ollama 1 (a local GPU serialises anyway), fake unlimited. Override with
`KOLM_CODE_PROVIDER_CONCURRENCY=cerebras=1,deepseek=8`. Both adapters wrap
every request in `limiterFor(provider).run`. 429 and 5xx retries use
`retryDelayMs`; a 429 also halves the limiter's concurrency for 60 s
(never below 1) and logs once via the returned reply `meta.throttled`.
`MAX_TRIES` becomes 4 for 429/529/503 and stays 2 for other 5xx.

### 3.4 Streaming for long outputs (`openai-compatible.js`)

When `maxTokens > 8000` or `config.stream === true`, request `stream: true`
and assemble the text and tool-call deltas (OpenAI delta format, including
the `tool_calls[].function.arguments` fragments). Keep the non-streaming path
for everything else. Usage comes from the final chunk (`stream_options:
{ include_usage: true }`); if the endpoint omits usage, estimate from
`estimateTokens`-style char counts and mark `usage.estimated: true`.

**Empty-on-length retry (found live, check 2).** A thinking model (qwen3.8
through Ollama's OpenAI layer) can spend the whole `max_tokens` on its
`reasoning` field and return `finish_reason: "length"` with empty content
and no tool calls. Such a reply is retried exactly once with
`min(max(4 * maxTokens, 8000), 16000)` tokens, in the same strict mode that
served the first call; `meta.emptyRetry = { from, to }` and
`meta.reasoningChars` record it, and the usage of both calls is summed so the
cost ledger sees every billed token. A reply that already carries content or
tool calls is never retried, and a call already at the cap is not retried.
The `reasoning` field (Ollama, OpenRouter) is read wherever
`reasoning_content` (DeepSeek, vLLM) is, on plain replies and on streams.

### 3.5 Registry (`registry.js`)

Add `tools` (boolean) and `textToolsHint` (string|null) to entries; add
`ollama:deepseek-coder-v2` with `tools:false`; add `openrouter:*` wildcard
entry (free/unknown pricing, `pricingVerified:false`) if absent.

Tests: `tests/ladder-text-tools.test.js` (new), `tests/ladder-providers.test.js`
(extend: fallback flip on 400, limiter honours Retry-After, streaming
assembly from a fake fetch that yields SSE chunks).

## 4. Coordination (owner ORCH; starts after CTX, CODER, PROV land)

### 4.1 Plan v2 (`planner.js`)

`PLAN_SCHEMA` gains:

```
complexity: 'trivial'|'small'|'medium'|'large'
subtasks: [{ id, title, summary, files: string[], acceptance: string[], depends_on: string[] }]  // optional; [] or absent means one unit of work
integration_test: string|null   // command that proves the whole task, when subtasks exist
```

Prompt rules for the planner: split into subtasks ONLY when the pieces touch
disjoint files and can be verified separately; never more than 4; a subtask
must not need another's uncommitted work unless listed in `depends_on`. The
opening message renders `briefToPrompt(brief)` where the orchestrator has
attached `brief.retrieval`, `brief.instructions` and `brief.lessons` (section 4.6);
raise the planner's `BRIEF_CAP` from 10 KB to 32 KB. `l3-semantic.js` and
`l4-judge.js` keep their caps (the reviewer and judge see the diff, not the
snippets; ORCH may strip `brief.retrieval` before calling them).

### 4.2 Adaptive routing: `src/ladder/routing.js`

```js
export function routeRun({ plan, config, coderRungs }) -> { startRung, candidates, repairRounds, reason }
```

Policy (`config.routing`: `'auto'` default, `'fixed'` = today's behaviour):
trivial -> rung 0, 1 candidate, 1 repair round; small -> rung 0, configured
candidates; medium -> rung 0, configured candidates; large -> rung
`min(1, last)`, configured candidates + 1 (cap 4). `risk: 'high'` never lowers
candidates below 2. Emit `run.route` with the decision and reason.

### 4.3 Coach: `src/ladder/coach.js`

```js
export async function coachRepair({ orchestrator, plan, feedback, diffText, transcriptTail, config, store, runId, candidateId, signal }) -> { hints: string[], costUsd }
```

One cheap orchestrator call (`effort: 'low'`, `maxTokens` 600, JSON schema
`{ hints: string[] }`) that turns raw stage output (failing test names,
lint lines, judge blocking items) plus the candidate's diff (capped 12 KB)
and the last 3 tool results into at most 5 concrete instructions ("the
failing assertion expects 5400; your UNITS map is missing h", "do not edit
tests/x; the test is correct"). Skipped when the orchestrator ref equals the
coder ref, when `config.coach === false`, or on budget. Hints go to
`runRepair({ hints })` (section 2.4). Emit `repair.coach` `{ candidateId, round,
hints, costUsd }`. A coach failure is never fatal: log, emit, continue with
raw feedback.

**Coder that never calls a tool (found live, check 5).** Run
`r_20260909_b4d9777c` (bench task `fix-duration-units`, coder
`ollama:mistral:latest`, everything else `ollama:qwen3.8:latest`) failed in 56.3s
without a single tool call: both turns described the call in prose ("I will use
the `edit_file` function ... Here's the command to do that"), the nudge changed
nothing, the coach's correct hint changed nothing, and the same task passes in
about 40s with qwen3.8 as coder. A coordinator must not fail a run because one
worker is structurally unable to act, so there are two steps:

1. **Change the format, not the volume.** On the first idle turn of a session
   that has never called a tool while it was offered native tools, `runCoder`
   switches THAT session to the text tool protocol (section 3.1) - `coder.chat({
   textTools: true })` for every remaining turn, per session, never latched on
   the shared adapter instance - and replaces the nudge with one that spells the
   `<<<tool name="...">>> ... <<<end>>>` form out. A protocol written into the
   system prompt is a format many small models follow after their native
   function calling has silently produced nothing. Emits `coder.protocol`; the
   session's end state rides out on `runCoder().protocol` and on
   `candidate.diff.protocol`.
2. **Reassign work a worker cannot do.** When an attempt still ends
   `reason: 'no-tool-calls'`, the orchestrator hands its repair round to a
   fallback coder - the next rung, else the orchestrator model when its ref
   differs from the coder's - instead of asking the same model the same thing
   again. Emits `coder.fallback`, and the fallback ref is recorded as the
   attempt's `generatorRef` so `show` and the receipt name the model that
   actually wrote the diff. With no other model to give it to (one rung, and the
   orchestrator is that rung), the round runs where it runs today.

### 4.4 Subtask coordination: `src/ladder/subtasks.js` + `orchestrator.js`

When `plan.subtasks.length >= 2` and `config.subtasks !== false`:

1. Build the DAG (`topoLevels(subtasks)`; a cycle or unknown `depends_on` is a
   `bad-plan` error that falls back to single-unit mode with an event
   `subtasks.rejected`).
2. Level by level: every subtask in a level runs as its own ladder (the
   existing per-rung generate -> cascade -> repair -> escalate loop, factored
   into `runUnit({ unit, baseRef, ... })`), in parallel across the level (pool
   size `config.subtaskConcurrency`, default 2), with candidate ids
   `s<k>c<rung>a<attempt>` and worktree ids that include the subtask id.
3. Integration branch: `kolm/code/<runId>/integration`, created from the base.
   Each finished subtask winner is merged in dependency order with
   `driver.merge(integrationWt, winnerBranch)` (new driver method: local-git
   does `git merge --no-ff --no-edit`; fake applies the diff; cloudflare
   throws `unsupported`). On conflict: run one repair-style coder session in
   the integration worktree with the conflict markers as feedback (`stage:
   'integration'`), at most `repairTurns` turns; if it still fails, the run is
   FAIL with `stopReason: 'integration-conflict'`.
4. After the last merge, run the cascade once on the integration worktree
   (`l0`-`l2` always; `l3`/`l4` per the usual bands) with the whole plan's
   acceptance and `integration_test` as the test command. Its verdict is the
   run's verdict; the receipt's winner is the integration commit, and the
   receipt lists `subtasks: [{ id, winner, branch, sha }]`.
5. Events: `subtasks.plan` (the DAG), `subtask.start`/`subtask.end`
   `{ id, verdict, winner, costUsd }`, `integration.merge` `{ id, ok,
   conflicts: [paths] }`, `integration.verdict`.
6. Resume: subtask progress is reconstructed from `subtask.end` events; a
   killed run resumes at the first unfinished level.
7. Budget: `checkBudget` between subtasks; a budget stop leaves finished
   subtask branches in place and reports `stopReason: 'budget'` with the list.

Single-unit runs (the common case) must produce byte-identical events and
candidate ids to today, so the hardening tests keep passing.

**Scoped acceptance (found live, check 2).** A subtask is judged only
against criteria it can satisfy. `scopedAcceptance(plan, subtask)` uses the
subtask's own `acceptance` list when the planner filled it; otherwise it
keeps every parent criterion that names one of the subtask's files (path,
basename, or a specific stem as a whole word), keeps every criterion that
names no subtask file at all, and drops the ones that name only other
subtasks' files. A criterion that names both sides belongs to the subtask
whose file it names first (its subject), so "test/integration.test.js requires
../src/csv.js" is the test subtask's alone; a kept criterion that still names a
foreign file carries a note saying which part is this subtask's. Tokens shared
by two subtasks (a common basename such as `index.js`) decide nothing. When nothing survives, one criterion is
synthesised from the subtask title and files. The planner prompt asks for a
per-subtask list, and the L3/L4 criteria block carries a scope note whenever
`plan.subtask_id` is set, so a grader does not mark a criterion unmet for
work another subtask owns. Before this, every subtask reviewer failed the
other subtasks' criteria, each a0 candidate went to repair, and the repairs
widened diffs into sibling files, which then conflicted at integration.

**Merge conclusion (found live, check 2).** A merge git started is over only
once it is committed. `commit()` therefore always runs `git commit` while
`MERGE_HEAD` exists, even when `git status --porcelain` is empty, because a
repair that resolves a conflict by keeping HEAD's side leaves exactly that
state; the non-merge case is unchanged (a clean tree adds no commit and reports
the current HEAD). `mergeSubtask` commits the integration worktree as soon as a
repair's markers are gone - message `kolm code: merge <id> (conflict resolved)`
- and takes `integrationRef` from that commit, so the next subtask of the same
level starts from a concluded merge instead of being refused; the level-end
commit stays and is a no-op when nothing changed. A merge failure with NO
conflicted file is not repairable by a coder: no repair session runs, the
`integration.merge` event carries git's reason as `detail` (scrubbed, 500
chars; failures only, so a clean merge's payload is unchanged), and the run
stops with `stopReason: 'integration-conflict'` and an `integration.verdict`
FAIL whose reason is `merge of <id> failed: <detail>`. After the post-repair
commit the branch is verified against the history, not the tree:
`wt.isAncestor(ref)` (`git merge-base --is-ancestor <ref> HEAD`; the fake driver
answers from the refs its `merge()` recorded, and drivers without the method are
skipped) must say the subtask branch landed, otherwise the second
`integration.merge` event is `{ ok: false, repaired: true, detail: 'branch <ref>
is not in the integration history after repair' }` and the run stops. Before
this, a resolved-to-HEAD conflict skipped its commit, `MERGE_HEAD` survived, the
next level's merge was refused with no conflict list, the repair session had
nothing to repair, and the run reported `{ ok: true, repaired: true }` for a
branch it had never merged.

**Failure policy (found live, r_20260909_f02fbbb9).** Both subtasks of that run
FAILED, nothing merged, the cascade graded the untouched base and marked it bad
on `l0-meta`, and a resume would have called both failures finished work. Three
rules follow from it. First, resume reuses a subtask only when its last
`subtask.end` says PASS and carries a branch: the branch is what the integration
merge consumes, and a FAIL or a JUDGE is exactly the work
`resume --coder <stronger>` exists to redo. A reused subtask is silent as
before; a redone one emits `subtask.retry { id, index, priorVerdict }`
immediately before its `subtask.start`, and its own ladder still resumes from
the cached candidates and stage results the store holds, so a rung that already
wrote a diff is not paid for twice. Second, a subtask whose dependency did not
pass never runs: levels are topological, so every dependency is decided by the
time a level starts, and a subtask with a failed or skipped dependency emits
`subtask.skipped { id, index, because: [ids] }` (the direct dependencies it was
waiting on), gets a ledger entry `{ verdict: 'SKIPPED', winner: null, branch:
null, sha: null, costUsd: 0 }`, and gets no `subtask.start`/`subtask.end` pair at
all. Skipping is transitive by construction, because a skipped subtask is itself
a subtask that did not pass. Independent subtasks in the same or a later level
still run: their work is what a resume with a better coder builds on. Third, the
integration cascade does not judge a partial diff. The PASS survivors are still
merged and the level is still committed, so the integration branch carries the
partial work; but when any subtask did not PASS, no `integration` candidate is
drawn, no stage runs, and the run emits `integration.verdict { verdict: 'FAIL',
winner: null, reasons, subtasks }` with one ASCII reason per non-PASS subtask in
plan order (`subtask alpha failed`, `subtask beta skipped (depends on alpha)`,
capped at 6) and stops with `stopReason: 'subtask-failed'` unless budget or an
integration conflict already claimed the stop. The `integration-conflict` path is
unchanged. Fourth, a run that FINISHED is reopened rather than replayed unless
it PASSED: a stored FAIL or JUDGE puts the run row back to `running`, emits
`run.resume { reopened: true, priorVerdict, priorStopReason }` and carries on
through the normal resume flow, so `resume --coder <stronger>` recovers a whole
run and not only a subtask inside a killed one (a stored PASS still
short-circuits to `stopReason: 'already-finished'`). Nothing already coded is
coded again: every unit the reopened run runs offsets its candidate ids and the
`rung` field of its events by the number of rungs that unit already spent (per
id prefix, `''` for a single-unit run and `s<k>` for subtask k), which puts the
coders on the resume command line ABOVE the stored rungs, never re-codes a
candidate id whose worktree the hygiene step destroyed, and leaves the new
receipt listing both attempts' candidates over the whole run's ledger.

### 4.5 Lessons hook (`orchestrator.js`)

At `run.end`, when lessons are enabled: `extractLessonCandidates(events)`,
merge and save via section 1.5, emit `run.lessons { added: n }`. No model call.

### 4.6 Retrieval hook (`orchestrator.js`)

After `repoBrief`, call `retrieveContext(baseWt, task)`,
`loadInstructions(baseWt)` and `loadLessons(config, repoKey)`; attach them as
`brief.retrieval`, `brief.instructions`, `brief.lessons` so every consumer
that renders `briefToPrompt(brief)` (planner, coder, repair) sees them (section 1.3).
Pass a copy without `retrieval` to the reviewer and judge stages. Cache retrieval per
`(repoKey, sha, task hash)` in `context_cache` (existing table, kind
`retrieval`).

Tests: `tests/ladder-subtasks.test.js` (new: DAG, parallel levels with the
fake driver, conflict path, resume mid-level, budget stop), extend
`tests/ladder-orchestrator.test.js` (routing, coach wiring, lessons hook) and
`tests/ladder-e2e.test.js` (a 2-subtask run end to end on the fake driver
with a fake provider that plays both the planner and two coders). Add
`merge()` to `worktree/fake.js` and `worktree/local-git.js` (ORCH owns both
for this wave) with tests in `tests/ladder-worktree.test.js`.

### 4.7 Events added by this wave (all additive; payload shapes are the contract)

| event | payload | when |
|---|---|---|
| `context.pack` | `{instructionsBytes, instructionFiles: string[], lessons, retrievalFiles, retrievalBytes, retrievalCached}` | once per run after the three context hooks, zeros where a hook degraded |
| `context.warn` | `{kind, message}` | a context hook failed; the run goes on without it |
| `run.resume` | `{runId, mode, task, repoPath, coders}`, plus `{reopened: true, priorVerdict, priorStopReason}` when a finished FAIL or JUDGE run is reopened (section 4.4). A resume of a finished PASS still emits `{runId, mode, alreadyFinished: true, winner}` and nothing else | a resume starts, in place of `run.start` |
| `run.route` | `{complexity, risk, routing, startRung, candidates, repairRounds, reason}` | after the plan, before the first candidate |
| `run.lessons` | `{added, total}` | at run end when lessons are on |
| `repair.coach` | `{candidateId, round, hints: string[], costUsd}` | only when the coach produced at least one hint |
| `subtasks.plan` | `{levels: string[][], subtasks: [{id, title, files, dependsOn}], concurrency}` | the plan was split and validated |
| `subtasks.rejected` | `{reason, count}` | the plan carried subtasks the validator or the driver refused |
| `subtask.retry` | `{id, index, priorVerdict}` | a resume is redoing a subtask that last ended FAIL or JUDGE; emitted immediately before its `subtask.start` |
| `subtask.start` | `{id, index, title, files, dependsOn}` | a subtask's own ladder begins |
| `subtask.end` | `{id, index, verdict, winner, branch, sha, costUsd}` | a subtask's ladder ended |
| `subtask.skipped` | `{id, index, because: string[]}` | a dependency did not pass, so this subtask never ran; `because` lists its direct failed or skipped dependencies, and no `subtask.start`/`subtask.end` pair follows |
| `integration.merge` | `{id, ok, conflicts: string[]}` | one subtask branch merged into the integration worktree |
| `integration.verdict` | `{verdict, winner: 'integration', reasons: string[], subtasks: string[]}` | the merged tree passed or failed the ladder. When any subtask did not PASS, no stage runs at all: the event carries `verdict: 'FAIL'`, `winner: null` and one reason per non-PASS subtask in plan order (capped at 6), because judging the merged survivors would grade a change nobody proposed (section 4.4) |
| `coder.finish_rejected` | `{candidateId, turn, reason: 'with-reads' | 'with-failures', count}` | a premature finish was answered instead of obeyed (at most 3 per session) |
| `coder.fallback` | `{candidateId, from, to, why: 'no-tool-calls'}` | a repair round was handed to another model because the attempt it repairs never called a tool (section 4.3) |
| `repair.stuck` | `{unit: number\|null, round, criteria: number[], candidates: string[]}` | a repair round left every candidate blocked by the same acceptance criteria (`criteria`, 0-based) or produced no change in a candidate's failure (`candidates`); section 4.8 |
| `plan.criterion` | `{unit, round, index, verdict: 'retire' \| 'rewrite', reason, from, to, costUsd}` | the plan audit changed one acceptance criterion for the rest of that unit; never emitted for `keep` (section 4.8) |

`run.budget` gains `where: 'subtask'` on the split path. `coder.self_review`
gains `emptyDiff: true` when the review was the empty-diff nudge.
`candidate.diff` gains `protocol: 'text' | 'native'`, the protocol the coder
session ended in (section 4.3).

### 4.8 Plan audit on stuck criteria: `src/ladder/audit.js`

```js
export async function auditCriteria({ orchestrator, plan, criteria, task, diffText, fileFacts, config, store, runId, unit, round, budgetExhausted, signal }) -> { plan, changed, changes, verdicts, costUsd, skipped }
export function fileFacts(text) -> { bytes, lines, crlf, lf, bom, trailingNewline } | null
export async function fileFactsFor(wt, paths, { baseRef }) -> [{ path, facts, base }]
export function stuckAnalysis({ before, after }) -> { criteria: number[], candidates: string[] }
```

**A criterion no patch can satisfy (found live, check 6).** Run
`r_20260909_f02fbbb9` (bench task `fix-inventory-total`, every model
`ollama:qwen3.8:latest`) failed a subtask whose diff was correct from the first
attempt. The task text claimed two files used CRLF line endings; they are
LF-only (23 LF lines, 0 CRLF). The planner turned the claim into acceptance
criterion #2, and L3 rejected all three candidates with the identical reason -
`acceptance #2 not met: Both src/inventory.js and tests/inventory.test.js use
CRLF (\r\n) line endings throughout - no bare LF introduced.` - before and after
two repair rounds. Six coder sessions and six reviewer calls were spent on a
criterion no patch could ever meet, and nothing in the run questioned the plan.
A repair round assumes the plan is right and the patch is wrong; a coordinator
has to be able to test that assumption.

1. **Detection (deterministic, in the repair loop).** After a repair round's
   `settle()` returns rows and no winner was picked, compare the failing rows
   with the rows the round started from. A criterion index `k` is STUCK when
   every failing row of this round names `k`, every failing row of the previous
   settle named `k`, and nothing PASSed in either - one candidate that got past
   `k` proves the criterion is satisfiable and the patches were the problem. A
   candidate made NO PROGRESS when its failure signature (rejecting stage,
   acceptance indices, non-acceptance reasons, whitespace-collapsed) is the one
   it already had. Indices come from a pure parser over the graders' own
   wording: L3's `acceptance #<k> not met: <text>` (scoped-acceptance
   parenthetical and all) and L4's `Unmet acceptance: #<k> <text>; ...`; a
   reason that names no criterion (`judge blocked: spec_compliance`) contributes
   none, which is what keeps an unrelated failure from looking stuck. The
   graders number criteria from 1 and the events from 0, so `acceptance #2` is
   `criteria: [1]`. Emit `repair.stuck` once per round when either set is
   non-empty; audit only when `criteria` is non-empty.
2. **The audit (one bounded call).** `auditCriteria` mirrors the coach (4.3):
   one orchestrator call, `effort: 'low'`, `maxTokens` 800, JSON schema
   `{ verdicts: [{ index, verdict: 'keep' | 'retire' | 'rewrite', reason,
   rewrite? }] }`, `modelTimeoutMs(config)`, skipped when `config.audit ===
   false`, on budget, or with no armed orchestrator. Unlike the coach it is NOT
   skipped when the orchestrator ref equals the coder ref: the question is about
   the plan, not a second opinion on code. The prompt carries the task text, the
   plan, each stuck criterion with its index, the best failing candidate's diff
   (capped 8 KB) and the file facts below. `retire` is for a criterion the
   repository already contradicts that the task does not ask to change ("keep X"
   where X was never true); `rewrite` is for one checkable only in different
   words; `keep` is anything a coder could still do.
3. **File facts (measured, not reported).** `fileFacts(text)` counts bytes,
   lines, `\r\n` pairs, bare `\n`, a BOM and a trailing newline; `fileFactsFor`
   reads each path with `wt.readFile` and the same path in the base with
   `wt.readFileAt(ref, rel)` (new driver method: local-git runs `git show
   <ref>:<rel>` and answers null on failure, the fake driver answers from the
   files it was created with, cloudflare has none and is skipped). Paths come
   from the criterion text by shape (`[\w./-]+\.[a-z]{1,5}`), and a path the
   worktree does not have contributes no line - which is what keeps a wrong
   guess to the cost of one read. Rendered as
   `src/inventory.js: 23 lines, 0 CRLF, 23 LF, no BOM, trailing newline`.
4. **After a retire or a rewrite: re-settle, do not re-code.** The amendment
   produces a NEW plan object for the rest of the unit (the stored plan is never
   mutated; `plan.criterion` is the record) and the already-written diffs are run
   through `settle()` again with no coder session: L0-L2 stay cached, L3 is
   re-run because its cached verdict attested to criteria that no longer exist,
   L4 is uncached anyway. A winner there ends the unit; otherwise the repair loop
   continues with the amended plan, which every later repair, the coach and the
   judge also see. Candidates that made no progress while a criterion stayed
   stuck are not repaired again (if that leaves nothing to repair, the round
   breaks to the escalation path exactly as `repaired.length === 0` does).

Four rules bound the damage: the audit is never fatal (a failed call, bad JSON
or an abort means every verdict is `keep`); at most one audit per unit per run,
so a wrong `keep` cannot loop and a wrong `retire` cannot cascade; only the
criteria the detector named can be touched; and the last surviving criterion is
never retired, because a plan with nothing to attest to passes every candidate
vacuously. Costs go on the ledger with `stage: 'audit'`. Resume replays the
`plan.criterion` events of the unit before its first rung, so a resumed run is
judged against the plan the audit left behind.

Limitations: the amendment is unit-scoped - a subtask's retire does not
propagate to the parent plan or to the integration cascade, which are judged
against the plan as written; and the detector needs at least one repair round,
so a run configured with `repairRounds: 0` never audits anything.

## 5. Benchmark and operator surface (owner BENCH; parallel with ORCH)

### 5.1 Bundled task set: `tests/fixtures/ladder-bench/<task-id>/`

Each task is a directory with `task.json` `{ id, title, text, language,
expect: { verdict: 'PASS', changedFiles: [...], hiddenTest: 'tests/hidden.test.js' } }`,
a `repo/` tree (files to seed a fresh git repository at bench time), and
`hidden/` files copied in AFTER the run to judge the result independently of
the tests the coder saw or wrote. At least these tasks (all `node --test`,
plain JS, no dependencies, each < 25 files):

1. `fix-duration-units` (bug fix, one file, existing tests)
2. `add-cli-json-flag` (feature, two files + README)
3. `crlf-edit` (edit inside CRLF files, must keep CRLF)
4. `multi-file-rename` (rename a function across 4 files)
5. `write-tests-for-untested` (repo with no tests; the coder must add them and a test script)
6. `refactor-extract-module` (move 3 functions into a new module, keep exports)
7. `two-subtask-feature` (two independent modules + an integration test; the plan should split)
8. `flaky-baseline` (one pre-existing failing test the coder must not "fix")
9. `forbidden-path-trap` (the obvious fix edits a lockfile-like generated file; the right fix edits the source)
10. `python-fix-if-available` (pytest task; skipped with a reason when `python`/`pytest` is missing)

### 5.2 Runner: `src/ladder/bench/index.js`

```js
export async function listBenchTasks(opts) -> [{ id, title, language, dir }]
export async function runBench({ tasks, config, store, driver, providers, onEvent, signal, workDir, concurrency }) -> {
  version: 1, startedAt, finishedAt, ladder: { coders, reviewer, orchestrator, judge }, mode,
  results: [{ id, verdict, hiddenPass: boolean|null, costUsd, durationMs, rungReached, candidates, turns, escalations, repairs, subtasks, runId, reason }],
  summary: { tasks, pass, hiddenPass, passRate, costUsd, durationMs, escalations, repairs }
}
export function formatBenchTable(result) -> string   // ASCII table + summary line
export function compareBench(a, b) -> string          // per-task delta table
```

For each task: seed a fresh git repo in `workDir/<id>` (`git init`, commit
`bench base`), run `runTask` with the given providers/config, then copy
`hidden/` over the WINNER's worktree (or a fresh checkout of the winner
branch), run the hidden test command, and record `hiddenPass`. Results are
written to `<dataDir>/code/bench/<ts>.json`. The store is the normal ladder
store (runs are real and show in `kolm code status`).

### 5.3 CLI (`cli/kolm-code.mjs`)

- `kolm code bench [--tasks a,b] [--concurrency N] [--json] [--compare <file>]`
  with the same model flags as `run`; prints the table, the ladder used, the
  file written; exit 0 when every non-skipped task passes hidden, 2 otherwise.
- `kolm code lessons <repo> [--clear]` prints or clears the repo's lessons.
- `kolm code show <runId>` renders subtasks (id, verdict, winner, sha) and
  the integration verdict when present; `--events` already prints everything.
- `kolm code doctor` prints the routing mode, provider concurrency, text-tool
  refs and whether lessons/self-review/coach are on.

### 5.4 Config (`src/ladder/config.js`) and docs

Register every section 7 variable with defaults, validation and `configForModels`
support; `index.js` re-exports the new modules (`retrieveContext`,
`loadInstructions`, lessons functions, `runBench`, `routeRun`, text-tool
helpers). `docs/KOLM_CODE.md` gains sections: Coordination (subtasks,
integration, routing, coach), Context (retrieval, instructions, lessons),
Runtime (compaction, self-review, text tools, limiter), Benchmark (how to run,
how to read, how to add a task), and the section 7 table. Regenerate the five
matrices (`node scripts/build-backend-atomic-deep-dive.mjs --summary`,
`build-cli-command-matrix.cjs`, `build-artifact-matrix.cjs`,
`build-bench-harness-matrix.cjs`, `build-otel-matrix.cjs`) and run the pin
tests wave602/939/948/950/951.

Tests: `tests/ladder-bench.test.js` (new: task loading, seeding, hidden
verdict, table/compare rendering, all on the fake driver + fake provider with
a scripted "coder" that applies a known-good diff), extend
`tests/ladder-cli.test.js` and `tests/ladder-config.test.js`.

## 6. Live acceptance (run by the orchestrating session, not by agents)

On local Ollama (`ollama:qwen3.8:latest` everywhere, `ollama:deepseek-coder-v2`
as rung 0 for the text-protocol check):

1. `kolm code bench` full set: expect >= 8/9 hidden pass on the JS tasks.
2. `two-subtask-feature` shows `subtasks.plan`, two `subtask.end`, one
   `integration.verdict`; the receipt lists both subtasks.
3. rung 0 `deepseek-coder-v2` codes through the text protocol (no 400, no
   escalation) on `fix-duration-units`.
4. A `--coder-turns 40` run on `multi-file-rename` shows at least one
   `coder.compact` event with a 12k token budget (`KOLM_CODE_CONTEXT_TOKENS=12000`).
5. A repair round shows `repair.coach` hints that name the failing assertion.
6. A second run on the same fixture repo carries the earlier lessons: the
   `context.pack` event reports `lessons >= 1` and `show` prints them on its
   `context` line; the coder's opening therefore contains
   `LESSONS FROM EARLIER RUNS` (the transcript itself is not stored, the pack
   event is the evidence).
7. Kill a 2-subtask run after the first `subtask.end`; `resume` finishes it.

### 6.1 Evidence (2026-09-09, local Ollama, every model `ollama:qwen3.8:latest` unless noted)

| Check | Run | Result |
| --- | --- | --- |
| 1 | bench set `2026-09-09T08-53-10-546Z` | 10/10 PASS, 10/10 hidden tests, 0 escalations, 6 repairs (all on `two-subtask-feature`), 32.5 min |
| 2 | `r_20260909_6a5142fd` | `two-subtask-feature`: `subtasks.plan`, two `subtask.end`, `integration.verdict` PASS, receipt lists both subtasks; found the empty-on-length reviewer reply, the foreign-subject criterion and the dangling MERGE_HEAD (all fixed) |
| 3 | `r_20260909_18fc043b` | rung 0 `ollama:deepseek-coder-v2` through the text protocol: no 400, no escalation, PASS |
| 4 | `r_20260909_e2721302` | `multi-file-rename` under `KOLM_CODE_CONTEXT_TOKENS=6000`: `coder.compact` fired, PASS, hidden test passed |
| 5 | `r_20260909_b4d9777c` | `fix-duration-units` with coder `ollama:mistral`: `repair.coach` hints named the failing assertion; mistral described its tool calls in prose, which led to `coder.protocol` and `coder.fallback` (run itself FAIL, by design of the check) |
| 6 | `r_20260909_f02fbbb9` then `r_20260909_b1b13f37` | lessons packed (`lessons=3`, then 8); the first run burned 19 candidates, 252 coder turns and 12 repair rounds on a false CRLF criterion and failed; the rerun with the audit reported `repair.stuck` after round 1, retired the criterion with a reason citing the base file facts (0 CRLF / 27 LF), re-settled and passed with 6 candidates, 97 turns, 1 repair round |
| 7 | `r_20260909_7159a980` then `r_20260909_5e45755e` | killed after the first `subtask.end`, resumed to PASS with all three merges `ok=true`; the first rerun left the carried-over subtask's worktrees on disk (fixed: a resume adopts them), the second left only the integration worktree and marked all nine subtask rows destroyed |

Found live and fixed on top of the checks above: a resumed run whose stored
verdict was FAIL used to be answered with `already-finished` (now reopened,
`run.resume { reopened: true }`, new rungs numbered after the stored ones);
a run that never judged an integration gained a phantom `integration` row
marked destroyed on the way out (now only rows that exist are marked).

## 7. New configuration

| Variable | Default | Meaning |
|---|---|---|
| `KOLM_CODE_ROUTING` | `auto` | `auto` sizes candidates/start rung from plan complexity; `fixed` keeps the configured shape |
| `KOLM_CODE_SUBTASKS` | `1` | allow the planner to split into subtasks with parallel ladders and integration |
| `KOLM_CODE_SUBTASK_CONCURRENCY` | `2` | parallel subtasks per DAG level |
| `KOLM_CODE_COACH` | `1` | orchestrator writes repair hints |
| `KOLM_CODE_AUDIT` | `1` | orchestrator audits an acceptance criterion that blocked every candidate twice (section 4.8); `0` leaves the plan untouchable |
| `KOLM_CODE_SELF_REVIEW` | `1` | coder reviews its own diff before finish |
| `KOLM_CODE_LESSONS` | `1` | load/save per-repo lessons |
| `KOLM_CODE_CONTEXT_TOKENS` | (model ctx) | hard cap for the coder transcript budget |
| `KOLM_CODE_TEXT_TOOLS` | `` | comma list of refs forced onto the text tool protocol |
| `KOLM_CODE_PROVIDER_CONCURRENCY` | per-provider defaults | `name=n,...` |
| `KOLM_CODE_STREAM` | `0` | force streaming on OpenAI-compatible endpoints |
| `KOLM_CODE_RETRIEVAL_BYTES` | `16384` | retrieval snippet budget |
| `KOLM_CODE_BENCH_DIR` | `<dataDir>/code/bench` | bench result files |

## 8. Ownership

| Owner | Files (create or edit only these) |
|---|---|
| CTX | `src/ladder/context/retrieve.js`, `src/ladder/context/instructions.js`, `src/ladder/context/repo-brief.js`, `src/ladder/context/detect.js`, `src/ladder/lessons.js`, `tests/ladder-context.test.js`, `tests/ladder-lessons.test.js` |
| CODER | `src/ladder/coder.js`, `src/ladder/transcript.js`, `tests/ladder-transcript.test.js`, `tests/ladder-coder.test.js` |
| PROV | `src/ladder/providers/text-tools.js`, `src/ladder/providers/limiter.js`, `src/ladder/providers/openai-compatible.js`, `src/ladder/providers/anthropic.js`, `src/ladder/providers/registry.js`, `tests/ladder-text-tools.test.js`, `tests/ladder-providers.test.js` |
| ORCH | `src/ladder/orchestrator.js`, `src/ladder/planner.js`, `src/ladder/subtasks.js`, `src/ladder/routing.js`, `src/ladder/coach.js`, `src/ladder/audit.js`, `tests/ladder-audit.test.js`, `src/ladder/store.js`, `src/ladder/receipt.js`, `src/ladder/worktree/fake.js`, `src/ladder/worktree/local-git.js`, `src/ladder/worktree/cloudflare-sandbox.js`, `src/ladder/stages/index.js`, `tests/ladder-subtasks.test.js`, `tests/ladder-orchestrator.test.js`, `tests/ladder-e2e.test.js`, `tests/ladder-worktree.test.js`, `tests/ladder-receipt.test.js`, `tests/ladder-store.test.js` |
| BENCH | `src/ladder/bench/**`, `tests/fixtures/ladder-bench/**`, `src/ladder/config.js`, `src/ladder/index.js`, `cli/kolm-code.mjs`, `docs/KOLM_CODE.md`, `tests/ladder-bench.test.js`, `tests/ladder-cli.test.js`, `tests/ladder-config.test.js`, generated matrices under `docs/` |

`tests/ladder-hardening.test.js` is frozen: nobody edits it, and it must pass
at the end of every owner's work.
