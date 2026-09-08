# kolm code — confidence-cascade coding agent (build spec, 2026-09-09)

Status: BUILD SPEC (binding for the implementation wave). Owner: kolm.
Derived from an exhaustive review of publu/ladder (confidence cascade over
a cheap->expensive verifier stack) and translated into a coding agent that
kolm owns end to end. Where ladder grades documents, kolm code grades
*patches*; where ladder has one judge, kolm code has an orchestrator
(Claude Fable 5.1) that plans, delegates generation to cheap coders
(Cerebras / DeepSeek V4 Flash / local Ollama), runs every candidate through
a versioned verification ladder, and only spends the expensive judge on the
unsure residue. Every run ends with a signed evidence receipt.

## 0. Non-negotiables (read before writing code)

* Language: JavaScript ESM (`"type":"module"`), Node >= 22.5 (uses `node:sqlite`).
  No TypeScript, no build step. Windows + Linux must both work (paths via
  `node:path`, processes via `child_process.spawn(cmd, {shell:true})`).
* Everything new lives under `src/ladder/`, `cli/kolm-code.mjs`,
  `deploy/kolm-code-sandbox/`, `tests/ladder-*.test.js`, `docs/KOLM_CODE.md`.
  The ONLY edits to existing files: (a) `cli/kolm.js` gets one `case 'code'`
  plus `'code'` in `COMPLETION_VERBS` plus one help line; (b) `package.json`
  gets `@anthropic-ai/sdk` bumped to the latest published 0.x (0.124.0 at
  spec time) and a `test:ladder` script. Nothing else.
* Fail loud. A rung that is not armed (missing key / unreachable) throws
  `LadderError('not-armed', ...)` when *used*; it never silently degrades.
  `kolm code doctor` is the place that reports arming.
* No credential is ever read from files other than the explicit env vars
  listed in section 2 (plus the Truffle GitHub App key path when the user
  sets it). Never scan `~/.claude.json`, keychains, or other repos.
* Never `git add -A`. Stage explicit paths. Never commit; the orchestrating
  session commits.
* Brand: the viewer uses kolm's design tokens (`public/kolm-2026.css`
  `:root` values: room `#08090A`, panel `#0F1011`, register `#15171A`,
  ink `#F7F8F8` / `#D0D6E0` / `#8A8F98`, accent `#3FE5A0` for VERIFIED
  only, hairlines `rgba(255,255,255,.07)`, squared radii <= 3px, paper CTA
  `#F4F2EC` with dark ink). No glows, no gradients, no chartreuse.
* Never use the words "honest"/"honesty" anywhere (code, docs, UI, tests).
* Tests use `node:test` + `node:assert/strict`, file names
  `tests/ladder-<area>.test.js`, run by `node --test --test-concurrency=1 tests/ladder-*.test.js`.
  Every module ships with tests that exercise it end to end through its
  public API (not tautological unit stubs). Tests must not need network or
  keys; live paths are exercised by `kolm code doctor --live` and the smoke
  script in section 10.

## 1. Directory layout and ownership

```
src/ladder/
  errors.js                 LadderError(code, message, detail)          [A]
  config.js                 env parsing + defaults + arming report      [A]
  providers/
    index.js                createProvider(ref) / resolveRef            [A]
    registry.js             model catalog + pricing                     [A]
    anthropic.js            @anthropic-ai/sdk adapter                   [A]
    openai-compatible.js    fetch adapter: cerebras/deepseek/openrouter/ollama [A]
    fake.js                 scripted provider for tests                 [A]
  worktree/
    index.js                createDriver(kind, opts)                    [B]
    local-git.js            git worktree driver                         [B]
    cloudflare-sandbox.js   HTTP client for the Worker in deploy/       [B]
    fake.js                 in-memory driver for tests                  [B]
  store.js                  ladder.db (node:sqlite)                     [C]
  bands.js + bands.json     band() + thresholds                         [C]
  rubric.json               single source of truth for L3 + L4          [C]
  verdict.js                assembleVerdict / gates / funnel math       [C]
  context/
    repo-brief.js           decode-once repo brief cached per SHA       [C]
    baseline.js             baseline typecheck/lint/test per SHA        [C]
    detect.js               detect test/typecheck/lint commands         [C]
  stages/
    index.js                STAGES registry (level/version/workers/gate)[C]
    l0-meta.js              patch hygiene                               [C]
    l1-static.js            typecheck + lint delta                      [C]
    l2-dynamic.js           tests + changed-line coverage               [C]
    l3-semantic.js          cheap-model rubric review                   [C]
    l4-judge.js             Fable 5.1 judge on residue + calibration    [D]
  planner.js                orchestrator plan call                      [D]
  coder.js                  bounded tool-using coder loop in a worktree [D]
  orchestrator.js           generation ladder + cascade + repair + escalate [D]
  receipt.js                signed evidence receipt + verify            [D]
  eval.js                   layer-vs-judge calibration                  [D]
  index.js                  public API (runTask/resumeRun/...)          [D]
  viewer/
    server.js               `kolm code serve` http + SSE                [E]
    index.html              single-file viewer, kolm design system      [E]
  github/
    app-auth.js             GitHub App JWT -> installation token        [F]
    pr.js                   push branch + open PR (receipt in body)     [F]
  truffle/
    partner-task.js         truffle.channel-partner-task.v1 adapter     [F]
    worker.js               poll/claim/run/settle loop                  [F]
cli/kolm-code.mjs           `kolm code ...` verbs                       [E]
deploy/kolm-code-sandbox/   Cloudflare Worker + Sandbox container       [B]
tests/ladder-*.test.js      per owner                                   [A-F]
docs/KOLM_CODE.md           operator docs                               [E]
```

Owners A–F are disjoint. Cross-owner contracts are the interfaces below;
build against them, and against `fake.js` implementations, never against a
sibling's unfinished code.

## 2. Configuration (`src/ladder/config.js`)

`loadConfig(env = process.env)` returns a frozen object. Every key below is
an env var; defaults in brackets.

Orchestration
* `ANTHROPIC_API_KEY` — arms `anthropic:*` refs.
* `KOLM_CODE_ORCHESTRATOR` [`anthropic:claude-fable-5-1`] planner + judge model ref.
* `KOLM_CODE_JUDGE` [same as orchestrator] judge model ref (may be `anthropic:claude-opus-5`).
* `KOLM_CODE_MODE` [`cascade`] `cascade` | `direct`. `direct` = generate
  with the orchestrator model only, skip L3, judge everything. Exists so the
  cascade can be measured against the simpler alternative (`kolm code funnel --compare`).

Cheap rungs (each armed only when its key is present / endpoint reachable)
* `CEREBRAS_API_KEY` -> `cerebras:*` at `https://api.cerebras.ai/v1`.
* `DEEPSEEK_API_KEY` -> `deepseek:*` at `https://api.deepseek.com/v1`.
* `OPENROUTER_API_KEY` -> `openrouter:*` at `https://openrouter.ai/api/v1`.
* `KOLM_CODE_OLLAMA_URL` [`http://127.0.0.1:11434/v1`] -> `ollama:*` (no key; armed if `GET /api/tags` answers within 1500 ms; doctor probes it, providers do not).
* `KOLM_CODE_CODER_LADDER` [`cerebras:qwen-3.8-27b,deepseek:deepseek-v4-flash,anthropic:claude-opus-5`]
  escalation order for generation. Unarmed entries are dropped at load time
  with a warning in the arming report; if none remain the orchestrator model
  is the sole coder.
* `KOLM_CODE_REVIEWER_LADDER` [`ollama:qwen3.8:latest,cerebras:gpt-oss-120b,deepseek:deepseek-v4-flash`]
  L3 reviewers, first armed entry is used; if none armed L3 returns `skip`.

Budgets and shape
* `KOLM_CODE_CANDIDATES` [3], `KOLM_CODE_CODER_TURNS` [24],
  `KOLM_CODE_REPAIR_TURNS` [2], `KOLM_CODE_MAX_COST_USD` [5],
  `KOLM_CODE_CALIBRATION_RATE` [0.1], `KOLM_CODE_STAGE_TIMEOUT_MS` [600000],
  `KOLM_CODE_KEEP_WORKTREES` [`losers-destroyed`] | `all` | `none`.

Worktrees
* `KOLM_CODE_DRIVER` [`local`] `local` | `cloudflare` | `fake`.
* `KOLM_DATA_DIR` [`~/.kolm`] -> db at `$KOLM_DATA_DIR/code/ladder.db`,
  local worktrees at `$KOLM_DATA_DIR/code/worktrees/<runId>/<candidateId>`.
  Override db with `KOLM_CODE_DB_PATH`.
* `KOLM_CODE_SANDBOX_URL`, `KOLM_CODE_SANDBOX_TOKEN` — arm the cloudflare driver.

GitHub (env-gated, F)
* `KOLM_GITHUB_APP_ID`, `KOLM_GITHUB_APP_INSTALLATION_ID`,
  `KOLM_GITHUB_APP_PRIVATE_KEY_PATH` [`~/.truffle/truffle-pr-authority.private-key.pem` if that file exists, else unset].
  All three required to arm `--pr`.

Truffle (env-gated, F)
* `KOLM_TRUFFLE_BASE_URL`, `KOLM_TRUFFLE_PARTNER_TOKEN`, `KOLM_TRUFFLE_HARNESS_ID` [`kolm-code`].

`armingReport(config)` returns
`{ rungs:[{id, kind:'orchestrator'|'judge'|'coder'|'reviewer'|'driver'|'github'|'truffle', ref, armed:boolean, reason}] }`
used by doctor, by the viewer, and embedded in every receipt.

Model ref grammar: `<provider>:<model>`; split on the FIRST colon only
(`ollama:qwen3.8:latest` is provider `ollama`, model `qwen3.8:latest`).

## 3. Provider interface (`src/ladder/providers/*`) — owner A

```js
const p = createProvider('deepseek:deepseek-v4-flash', config);
p.id            // 'deepseek:deepseek-v4-flash'
p.provider      // 'deepseek'
p.model         // 'deepseek-v4-flash'
p.armed         // boolean
await p.chat({
  system,                       // string
  messages,                     // Anthropic-shaped: [{role:'user'|'assistant', content: string | Block[]}]
  tools,                        // [{name, description, input_schema}] optional
  jsonSchema,                   // {name, schema} optional -> response.json populated
  maxTokens = 8192, temperature = 0.2,
  effort,                       // 'low'|'medium'|'high'|'xhigh'|'max' (anthropic only; ignored elsewhere)
  cacheSystem = true,           // anthropic: cache_control on system block
  signal, timeoutMs
}) -> {
  text,                         // concatenated text blocks ('' if none)
  json,                         // parsed object when jsonSchema given, else null
  toolCalls,                    // [{id, name, input}]
  content,                      // raw assistant blocks, Anthropic-shaped, for appending to history
  stopReason,                   // 'end_turn'|'tool_use'|'max_tokens'|'refusal'|'stop'
  usage: {input, output, cacheRead, cacheWrite},
  costUsd, model, provider, latencyMs
}
```
Block shapes (unified, Anthropic-native): `{type:'text',text}`,
`{type:'tool_use',id,name,input}`, `{type:'tool_result',tool_use_id,content}`.
The OpenAI-compatible adapter translates both directions (tool_use ->
`tool_calls`, tool_result -> `role:'tool'` messages).

Anthropic adapter rules (from the claude-api skill; do not improvise):
* `import Anthropic from '@anthropic-ai/sdk'`; `client.messages.create`, or
  `client.messages.stream(...).finalMessage()` when `maxTokens > 16000`.
* Never send `thinking.budget_tokens`. Omit `thinking` (Fable 5.1 / Opus 5
  are adaptive by default). Pass `output_config: { effort }` when given.
* Never send `tool_choice` `any`/`tool` (400 on Fable 5.1). For JSON use
  `output_config: { format: { type: 'json_schema', schema } }` and parse
  the text; fall back to extracting the first `{...}` block if parsing fails
  and set `json=null` when that also fails (caller decides).
* `stop_reason === 'refusal'` -> return `stopReason:'refusal'`, never throw.
* `cacheSystem` -> `system: [{type:'text', text, cache_control:{type:'ephemeral'}}]`.
* Use SDK typed errors; map 429/529/5xx to retry (3 tries, expo backoff,
  jitter); map 400 to `LadderError('provider-bad-request')`.
* Cost: usage x pricing from registry (cache reads billed at the registry's
  `cacheRead` rate).

OpenAI-compatible adapter: `POST {base}/chat/completions` with
`Authorization: Bearer`, `response_format: {type:'json_schema', json_schema:{name, schema, strict:true}}`
when `jsonSchema` given (Cerebras and DeepSeek support it; on 400 retry once
without `strict`); `tools` in OpenAI shape; `usage.prompt_tokens/completion_tokens`
(+ `prompt_cache_hit_tokens` for DeepSeek). OpenRouter adds
`HTTP-Referer: https://kolm.ai`, `X-Title: kolm code`.

Registry (`registry.js`) — `describeModel(ref)` -> `{ref, provider, model, ctx, maxOut, priceIn, priceOut, cacheRead, roles:['coder'|'reviewer'|'judge'|'planner']}`.
Seed with: `anthropic:claude-fable-5-1` (1M/128K, $10/$50, cacheRead $0.25),
`anthropic:claude-opus-5` (1M/128K, $5/$25, cacheRead $0.50),
`anthropic:claude-sonnet-5`, `anthropic:claude-haiku-4-5`,
`deepseek:deepseek-v4-flash` (1M/384K, $0.44/$1.32 peak; `KOLM_CODE_DEEPSEEK_OFFPEAK=1` -> $0.22/$0.66),
`deepseek:deepseek-v4-pro`, `cerebras:qwen-3.8-27b`, `cerebras:kimi-k2.7-code`,
`cerebras:gpt-oss-120b`, `cerebras:gemma-4-31b`, `openrouter:deepseek/deepseek-v4-flash`,
`openrouter:deepseek/deepseek-v4-pro`, `ollama:*` (price 0). Unknown refs
are allowed with price 0 and `roles:['coder','reviewer']` and a `unknownPricing:true` flag.

`fake.js`: `createFakeProvider({script})` where `script` is an array of
responses or a function `(call, index) => response`; records every call in
`provider.calls` for assertions. Must support tool-call sequences so the
coder loop can be tested end to end.

## 4. Worktree interface (`src/ladder/worktree/*`) — owner B

```js
const driver = createDriver(config.driver, config); // 'local' | 'cloudflare' | 'fake'
const wt = await driver.create({ repoPath, repoUrl, baseRef, runId, candidateId, gitToken });
wt.id; wt.kind; wt.path;                    // path is local fs path (local) or container path (cloudflare)
await wt.exec(cmd, { cwd, timeoutMs = 300000, env, maxOutputBytes = 1<<20 })
  -> { stdout, stderr, exitCode, durationMs, timedOut, truncated }
await wt.readFile(rel) -> string | null
await wt.writeFile(rel, content)            // creates parents
await wt.deleteFile(rel)
await wt.listFiles({ glob, limit = 5000 }) -> string[]   // tracked + untracked, respects .gitignore
await wt.search(pattern, { glob, limit }) -> [{path, line, text}]  // git grep -n
await wt.diff() -> string                   // unified diff vs baseRef incl. untracked files
await wt.changedFiles() -> [{path, status:'A'|'M'|'D', added, deleted}]
await wt.applyPatch(diff) -> { ok, error }  // git apply --3way
await wt.commit(message) -> { sha }         // commits everything in this worktree only
await wt.baseSha() -> string
await wt.destroy()
await driver.destroyRun(runId)              // removes every worktree of a run
```
Local driver: one `git worktree add --detach <path> <baseSha>` per
candidate; branch created at commit time as `kolm/code/<runId>/<candidateId>`.
Refuse to operate when `repoPath` is not a git repo (report clearly).
Prune on destroy (`git worktree remove --force`).

Cloudflare driver: thin HTTP client for the Worker in
`deploy/kolm-code-sandbox` (section 8). Same interface; `exec` streams are
not required, buffered responses are fine.

Fake driver: in-memory files, `exec` dispatches to a scripted command
table (`{ 'npm test': {stdout, exitCode} }` or a function), diff computed
from the in-memory file map. Used by orchestrator/stage tests.

## 5. Store, bands, rubric, verdict, context, stages — owner C

### 5.1 `store.js` — `openStore(dbPath)` (node:sqlite, WAL, synchronous API)

```sql
runs(id TEXT PK, created_at, updated_at, status, task_json, repo_path, base_sha, driver, mode, config_json, winner_candidate, total_cost_usd, receipt_json)
candidates(id TEXT PK, run_id, idx INTEGER, attempt INTEGER, generator_ref, worktree_json, status, diff_sha256, diff_text, summary, created_at, updated_at)
results(run_id, candidate_id, stage, version INTEGER, verdict_json, created_at, PRIMARY KEY(run_id,candidate_id,stage,version))
judgments(id INTEGER PK, run_id, candidate_id, kind, verdict_json, created_at)      -- kind: residue|calibration|final
context_cache(repo_key, sha, kind, payload_json, created_at, PRIMARY KEY(repo_key,sha,kind))
events(id INTEGER PK, run_id, ts, type, payload_json)
cost_ledger(id INTEGER PK, run_id, candidate_id, stage, provider, model, input_tokens, output_tokens, cache_read, cost_usd, ts)
```
API: `createRun`, `updateRun`, `getRun`, `listRuns({limit, status})`,
`upsertCandidate`, `listCandidates(runId)`, `putResult({runId, candidateId, stage, version, verdict})`,
`getResult(runId, candidateId, stage, version)`,
`latestResults(runId, candidateId)` (max version per stage), `resultsFor(runId)`,
`putJudgment`, `listJudgments(runId)`, `cacheGet/cachePut`, `emit(runId, type, payload)`,
`eventsSince(runId, afterId)`, `addCost`, `costSummary(runId)`, `funnel({since, runIds})`, `close()`.
Verdict rows are never deleted; a re-run with a bumped stage version adds rows.
Candidate ids: `c<idx>a<attempt>` (e.g. `c1a0`, `c1a1`); run ids: `r_<yyyymmdd>_<8 hex>`.
Candidate rows are keyed by `id` = `<runId>/<candidateId>` internally; the API
takes `(runId, candidateId)` pairs so callers never build the composite key.

### 5.2 `bands.js` + `bands.json`
`band(value, key, lowIsBad)` -> `'bad'|'good'|'unsure'`, exactly ladder's
semantics: two thresholds `lo < hi`; when `lowIsBad` the good side is above
`hi`; otherwise below `lo`. `worst(decisions)` -> bad > unsure > good.
`loadBands(path?)` reads `bands.json` (overridable with `KOLM_CODE_BANDS_PATH`).
Ship defaults:
```json
{ "l0-meta":     { "patch_lines": {"lo":600,"hi":2500,"low_is_bad":false},
                   "files_touched": {"lo":15,"hi":40,"low_is_bad":false} },
  "l1-static":   { "new_type_errors": {"lo":0,"hi":3,"low_is_bad":false},
                   "new_lint_errors": {"lo":0,"hi":10,"low_is_bad":false} },
  "l2-dynamic":  { "newly_failing_tests": {"lo":0,"hi":0,"low_is_bad":false},
                   "changed_line_coverage": {"lo":0.4,"hi":0.8,"low_is_bad":true} },
  "l3-semantic": { "rubric_score": {"lo":0.55,"hi":0.85,"low_is_bad":true},
                   "reviewer_confidence": {"lo":0.4,"hi":0.75,"low_is_bad":true} } }
```
(`lo == hi` means no unsure band for that key; value <= lo is good, above is bad.)

### 5.3 `rubric.json`
Single source of truth consumed by L3 (cheap reviewer prompt) and L4
(judge prompt). Items with `id, weight, title, question, evidence` (what the
grader must look at), `blocking` (bool): `correctness`, `scope_discipline`,
`tests`, `no_placeholders`, `error_handling`, `style_consistency`,
`no_secrets`, `spec_compliance`, `reversibility`. `loadRubric()` validates
shape and weights sum to 1. `rubricToPrompt(rubric)` renders it.

### 5.4 `verdict.js`
`assembleVerdict(stageDecisions)` -> `{ verdict:'PASS'|'FAIL'|'JUDGE', reasons, unsure:[stage...], bad:[stage...] }`
Ladder semantics: any confident-bad -> FAIL; any unsure -> JUDGE; else PASS.
Gate: `flowsUp(prev)` = previous stage decision is `good`, `unsure` or `skip` (NOT
`bad`). Deliberate deviation from ladder's good-only gate: verification
rungs are cheap relative to the judge, so an unsure candidate keeps
accruing evidence for the judge instead of skipping straight to it.
`funnel(rows)` -> per-stage `{good, unsure, bad, skip}` counts, `clearedCheaply`
(fraction whose final verdict needed no judge), `judged`, `hardFail`.

### 5.5 Context (`context/*`)
* `detect.js`: `detectCommands(wt)` -> `{ test, typecheck, lint, coverage, language, packageManager }` by inspecting
  `package.json` scripts, `tsconfig.json`, `pyproject.toml`/`pytest.ini`, `go.mod`, `Cargo.toml`,
  `.eslintrc*`/`eslint.config.*`, `ruff.toml`. Commands are strings run via `wt.exec`.
  Node: test=`npm test --silent` (or the detected runner), typecheck=`npx tsc --noEmit -p .` when tsconfig exists,
  lint=`npx eslint . -f json` when eslint config exists, coverage via `c8`/`nyc`/`--experimental-test-coverage` if available.
  Python: `pytest -q`, `pyright`/`mypy`, `ruff check --output-format json`. Go: `go test ./...`, `go vet ./...`. Rust: `cargo test`, `cargo check`.
* `repo-brief.js`: `repoBrief(store, wt, sha)` cached by `(repoKey, sha, 'brief')`:
  file tree (depth-limited, ignoring vendored dirs), language histogram,
  manifests (package.json name/scripts/deps names only), README head (2 KB),
  detected commands, top-level exports per source file (regex `export (function|class|const) NAME`,
  `def NAME`, `func NAME`). Target <= 12 KB of text; `briefToPrompt(brief)` renders it.
* `baseline.js`: `baseline(store, wt, sha, commands)` cached by `(repoKey, sha, 'baseline')`:
  runs typecheck/lint/test once on the untouched base and stores
  `{ typecheck:{errors:[...ids]}, lint:{errors:[...ids]}, tests:{passed, failed, failing:[names]}, durations }`
  so stages compute deltas, never absolutes. Test-name parsing supports
  `node --test` (TAP / spec output), jest/vitest summary lines, pytest `FAILED name`, go `--- FAIL: Name`.

### 5.6 Stages (`stages/*`)
Each stage module exports `{ id, level, version, workers, gate, run }` where
`run({ candidate, wt, plan, brief, baseline, commands, bands, rubric, providers, config, store, signal })`
returns
```js
{ decision:'good'|'bad'|'unsure'|'skip', reasons:[string], hits:[{key, value, band}],
  metrics:{...}, artifacts:{...small, e.g. failing test names, first 4 KB of output}, costUsd, latencyMs, model? }
```
* `l0-meta` (v1, workers 8): diff non-empty (empty -> bad); `wt.diff()` parses; size bands; forbidden paths (plan.forbidden_paths + defaults `.env*`, `*.pem`, `node_modules/`, `.git/`); binary files -> bad; secret scan (AWS/Anthropic/OpenAI/GitHub token regexes + high-entropy 32+ char strings in added lines) -> bad; `TODO|FIXME|placeholder|not implemented` in added lines -> unsure with hit; changed JSON/YAML files must parse (else bad).
* `l1-static` (v1, workers 4): run typecheck + lint if detected; delta vs baseline; bands. No tools detected -> `skip` with reason.
* `l2-dynamic` (v1, workers 2): run tests; delta of newly failing (present in fail set but not in baseline fail set) -> band; tests newly passing that were failing in baseline are recorded (`fixed`). Coverage of changed lines when coverage command detected, else metric absent (band only on present metrics). Test command absent -> `skip`.
* `l3-semantic` (v1, workers 4): first armed reviewer from `KOLM_CODE_REVIEWER_LADDER` (passed in as `providers.reviewer`, may be null); prompt = rubric + plan + brief excerpt + diff (truncated to 60 KB with a note) + L0–L2 summaries; `jsonSchema` output
  `{ items:[{id, score:0..1, evidence}], overall:0..1, confidence:0..1, blocking:[id...] }`;
  decision from bands on `rubric_score=overall` and `reviewer_confidence`, and any `blocking` item with score < 0.3 -> bad. No reviewer -> `skip`.
`stages/index.js` exports `STAGES` in level order and `runCascade({ runId, candidates:[{candidateId, wt}], ...ctx })`
which runs levels sequentially, candidates in parallel within a level
(bounded by `workers`), honors `gate`, writes every result via `store.putResult`
with the stage `version`, emits `stage.result` events, and returns `{ perCandidate: Map<candidateId, {decisions:{stage->decision}, verdict}> }`.
Skipping an already-present `(candidate, stage, version)` row = resumability.
`l4-judge` is NOT part of `runCascade`; the orchestrator calls it on the residue.

## 6. Orchestrator, coder, judge, receipt, eval — owner D

### 6.1 `planner.js`
`plan({ provider, task, brief, config })` -> JSON (`jsonSchema`):
`{ summary, approach:[steps], files_to_touch:[paths], forbidden_paths:[], acceptance_criteria:[], test_plan, risk:'low'|'medium'|'high', estimated_patch_lines }`.
Uses `effort:'high'`. When the planner ref is unarmed -> `LadderError('not-armed')` (the CLI prints the doctor hint). Fake provider in tests.

### 6.2 `coder.js`
`generate({ provider, wt, task, plan, brief, commands, feedback, config, store, runId, candidateId, signal })`
Bounded tool loop (`KOLM_CODE_CODER_TURNS`) with tools:
`read_file{path}`, `write_file{path, content}`, `delete_file{path}`,
`list_files{glob}`, `search{pattern, glob}`, `run{cmd}` (allow-listed prefixes:
detected test/typecheck/lint commands, `node`, `npm`, `npx`, `python`, `pytest`,
`go`, `cargo`, `git diff`, `git status`; 120 s cap; output truncated to 16 KB),
`finish{summary}`. System prompt states the rubric, the plan, the acceptance
criteria, the rule "touch only what the task needs", and, on repair turns,
the exact failures (`feedback`: stage reasons + artifacts + judge
`repair_instructions`). Returns `{ diff, summary, turns, costUsd, stoppedBy:'finish'|'turns'|'cost'|'refusal' }`.
Cost accounting via `store.addCost` per call. Works with any provider
(Anthropic native tools; OpenAI-compatible tools).

### 6.3 `l4-judge.js`
`judge({ provider, candidate, wt, plan, brief, rubric, evidence, kind:'residue'|'calibration'|'final', config, store, runId })`
-> `{ verdict:'pass'|'fail'|'revise', rubric:[{id, score, evidence}], reasons, repair_instructions, confidence }`
(`jsonSchema`, `effort:'high'`, `cacheSystem`). Evidence = all lower-rung
results (latest versions) + diff (truncated 120 KB) + baseline summary.
Stored via `store.putJudgment`.

### 6.4 `orchestrator.js` — `runTask({ task, repoPath, repoUrl, baseRef='HEAD', options }, deps)`
`deps = { config, store, createProvider, driver, stages, logger, signal }` (all
injectable for tests). Algorithm:
1. Create run; create a scratch worktree `c0a0`, resolve `baseSha`; `brief` + `baseline` (cached per SHA); destroy scratch.
2. `plan` via orchestrator provider.
3. For rung r in coder ladder (direct mode: ladder = [orchestrator]):
   a. Spawn `KOLM_CODE_CANDIDATES` candidates in parallel worktrees (`c<i>a0`), `generate` each.
   b. `runCascade` over the attempt's candidates (L0..L3; L3 skipped in direct mode).
   c. `assembleVerdict` per candidate. PASS candidates go to step 4.
      For FAIL/JUDGE candidates, run up to `KOLM_CODE_REPAIR_TURNS` repairs:
      new attempt id `c<i>a<k>`, same worktree (state carried), `feedback` =
      reasons; re-run cascade (rows keyed by attempt id keep history).
   d. If at least one candidate is PASS or JUDGE after repairs -> step 4;
      else escalate to rung r+1 (`store.emit('escalate')`). Ladder exhausted -> run FAIL.
4. Judge: every JUDGE candidate -> `judge(kind:'residue')`. PASS candidates:
   a `KOLM_CODE_CALIBRATION_RATE` sample (deterministic by hash of run id +
   candidate id, so tests are stable) -> `judge(kind:'calibration')`; a
   calibration `fail` overrides PASS (and is flagged `calibration_disagreement`).
   `revise` -> one extra repair with `repair_instructions` -> cascade -> `judge(kind:'final')`.
   Direct mode: judge every candidate.
5. Winner among passing candidates: fewest unsure stages, then smallest patch
   lines, then lowest cost. `wt.commit()` on the winner -> branch
   `kolm/code/<runId>/<candidateId>`; losers destroyed per `KOLM_CODE_KEEP_WORKTREES`.
6. Cost guard: abort with `status:'budget-exceeded'` when the ledger passes
   `KOLM_CODE_MAX_COST_USD` (checked before every provider call).
7. `receipt.build(...)` -> signed; `updateRun({status:'passed'|'failed'|..., receipt_json})`;
   write `receipt.json` + `winner.diff` under `$KOLM_DATA_DIR/code/runs/<runId>/`.
8. `resumeRun(runId, deps)`: rebuild state from the store (candidates,
   latest results) and continue from the first incomplete step; existing
   `(candidate, stage, version)` rows are not recomputed.
Emit events for every transition (`run.start`, `plan`, `candidate.start`,
`candidate.diff`, `stage.result`, `repair`, `escalate`, `judge`, `winner`,
`receipt`, `run.end`) so the viewer can replay a run live.

### 6.5 `receipt.js`
`build({ run, candidates, results, judgments, cost, arming, timings })` ->
```json
{ "spec":"kolm-code-receipt-v1", "run_id":"", "created_at":"", "task":{...}, "repo":{"path_hash":"sha256","base_sha":""},
  "mode":"cascade", "winner":{"candidate":"","branch":"","commit":"","diff_sha256":"","patch_lines":0,"files":[]},
  "funnel":{"l0-meta":{"good":0,"unsure":0,"bad":0,"skip":0},"cleared_cheaply":0.0,"judged":0,"hard_fail":0},
  "candidates":[{"id":"","generator":"","attempt":0,"stages":{"l0-meta":{"decision":"","version":1}},"verdict":"","judge":null}],
  "models":{"orchestrator":"","judge":"","coders":[],"reviewers":[]},
  "cost":{"total_usd":0,"by_model":{}}, "arming":[], "timings":{},
  "signature_ed25519":{ } }
```
Canonical payload = stable-key JSON of the receipt without
`signature_ed25519` (implement `canonical()` locally: recursive sorted keys,
no whitespace, arrays in order; test it), signed with
`loadOrCreateDefaultSigner()` + `buildSignatureBlock()` from `src/ed25519.js`.
`verify(receipt)` -> `{ ok, key_fingerprint, reason }` uses `verifySignatureBlock`.
Repo path is hashed, never stored raw (receipts may be shared).

### 6.6 `eval.js`
`evalLayers(store, { runIds | since })` compares each stage's decision with
the judge's verdict where both exist (residue + calibration rows):
per stage `{ agree, falseGood (stage good, judge fail), falseBad, unsureRate }`
and proposes band adjustments (`suggested_bands`) by sweeping thresholds
over stored `hits` values to maximize agreement subject to falseGood == 0.
`kolm code bands --suggest` prints them; `--apply` writes a bands.json.

### 6.7 `index.js`
`export { runTask, resumeRun, evalLayers, openStore, loadConfig, armingReport, createProvider, createDriver, STAGES, buildReceipt, verifyReceipt, createDeps }` where `createDeps(config)` wires real providers/driver/store.

## 7. CLI + viewer + docs — owner E

`cli/kolm-code.mjs` exports `main(argv)`; `cli/kolm.js` gets
`case 'code': await withErrorContext('code', async () => (await import('./kolm-code.mjs')).main(rest)); break;`
next to `case 'agent'`, `'code'` appended to `COMPLETION_VERBS`, and a help
line `code      confidence-cascade coding agent (kolm code help)`.
Exit codes reuse kolm's: 0 ok, 1 bad args, 2 gate fail (run FAILED), 3
missing prereq (not armed), 4 execution, 5 not found.

Verbs (all accept `--json`):
* `kolm code run "<task>" [--repo <path>] [--base <ref>] [--driver local|cloudflare|fake] [--mode cascade|direct] [--candidates N] [--max-cost USD] [--pr] [--keep all|losers-destroyed|none]`
  streams progress lines (`[l2-dynamic] c1a0 good 0 newly failing (2.1s $0.000)`), prints the
  winner branch, receipt path, funnel and cost table; exit 2 on FAIL.
* `kolm code resume <run-id>`
* `kolm code status [<run-id>]` / `kolm code show <run-id>` (candidate x stage matrix, judge verdicts, cost)
* `kolm code funnel [--since <iso>] [--compare]` (`--compare` splits cascade vs direct runs: cost per passed run, judge share, wall time)
* `kolm code eval [--since <iso>]` / `kolm code bands [--suggest|--apply <file>]`
* `kolm code receipt verify <file>`
* `kolm code doctor [--live]` — table of every rung from `armingReport` plus: git version, node:sqlite availability, Ollama reachability, wrangler login state (`wrangler whoami` if wrangler on PATH), GitHub App vars, Truffle vars. `--live` sends a 1-token request to each armed LLM rung and reports latency/cost.
* `kolm code serve [--port 7431] [--open]`
* `kolm code truffle-worker` (delegates to F's worker)
* `kolm code help`

Viewer (`src/ladder/viewer/`): `server.js` (`node:http`, no express) serves
`index.html` at `/`, JSON at `/api/runs`, `/api/runs/:id` (run + candidates +
latest results + judgments + cost), `/api/funnel`, `/api/doctor`, and SSE at
`/api/runs/:id/events` (poll `eventsSince` every 750 ms). `index.html` is one
file, inline CSS/JS, no CDN, kolm tokens (section 0): runs list, per-run
funnel bars per level (good/unsure/bad/skip), candidate x stage matrix with
version badges, live event log, cost ledger by model, receipt verification
state (calls `/api/runs/:id/receipt/verify`). Keyboard: `j/k` runs, `r` refresh.

`docs/KOLM_CODE.md`: what it is, the ladder diagram, env table, `doctor`
output explained, cascade vs direct measurement, Cloudflare deploy steps,
GitHub App + Truffle arming, receipt format + offline verify.

## 8. Cloudflare Sandbox worktrees — owner B (`deploy/kolm-code-sandbox/`)

Mirror Truffle's package shape (`deploy/agent-home-sandbox` in the Truffle
repo: `@cloudflare/sandbox` 0.12.9, wrangler 4.128+, `containers` +
`durable_objects` bindings, `new_sqlite_classes` migration). Files:
`package.json`, `wrangler.jsonc`, `Dockerfile` (`FROM docker.io/cloudflare/sandbox:0.12.9`),
`src/worker.mjs`, `src/sandbox.mjs`, `src/api.mjs`, `test/*.test.mjs`, `README.md`.

* `export class KolmCodeSandbox extends Sandbox {}` bound as `SANDBOX`
  (container class `KolmCodeSandbox`, `instance_type: 'standard'`, `max_instances: 10`,
  `sleepAfter` 20 min). One container per run: `getSandbox(env.SANDBOX, runId)`.
* Auth: every request needs `Authorization: Bearer ${env.KOLM_CODE_SANDBOX_TOKEN}` (wrangler secret); constant-time compare; 401 otherwise.
* API (JSON):
  `POST /v1/runs/:runId/worktrees {repoUrl, baseRef, candidateId, gitToken?}` -> clones `--depth 50`
  into `/workspace/<runId>/base` once (idempotent; `gitToken` injected via `-c http.extraheader`, never persisted),
  then `git worktree add --detach /workspace/<runId>/<candidateId> <baseSha>`; returns `{id, path, baseSha}`.
  `POST /v1/runs/:runId/worktrees/:id/exec {cmd, cwd?, timeoutMs?, env?}` -> `{stdout, stderr, exitCode, durationMs, timedOut}` (cap output 1 MB, timeout via `timeout` binary).
  `GET /v1/runs/:runId/worktrees/:id/file?path=` -> `{content}` | 404,
  `PUT ... /file {path, content}`, `DELETE ... /file?path=`,
  `GET ... /files?glob=`, `GET ... /search?pattern=&glob=`, `GET ... /diff`, `GET ... /changed`,
  `POST ... /patch {diff}`, `POST ... /commit {message}` -> `{sha}`,
  `DELETE /v1/runs/:runId/worktrees/:id`, `DELETE /v1/runs/:runId` (destroys the container).
* Path safety: every `path`/`cwd` must resolve inside the worktree root; reject `..`, absolute paths, symlink escapes (`realpath` check via exec).
* Tests (`node --test`): the API layer is tested with a fake `Sandbox`
  (`{exec, readFile, writeFile, mkdir, deleteFile}` recorders) injected via
  `createApi({ getSandbox })`; auth, path safety, idempotent clone, patch
  round trip, destroy. `npm run build` = `wrangler deploy --dry-run --outdir .build` must succeed (no login needed).
* README: `wrangler login`, `wrangler secret put KOLM_CODE_SANDBOX_TOKEN`, `wrangler deploy`, then
  `KOLM_CODE_DRIVER=cloudflare KOLM_CODE_SANDBOX_URL=https://kolm-code-sandbox.<acct>.workers.dev KOLM_CODE_SANDBOX_TOKEN=...`.

## 9. GitHub App PR + Truffle adapter — owner F

* `github/app-auth.js`: `installationToken({ appId, installationId, privateKeyPem })`
  (RS256 JWT via `node:crypto`, 9-min exp, `POST /app/installations/:id/access_tokens`); cached until 5 min before expiry.
* `github/pr.js`: `openPullRequest({ repoPath, branch, baseBranch, title, body, remote='origin', token })`:
  parses `owner/repo` from the remote URL, pushes the branch with the token
  via `-c http.extraheader` (never written to config), `POST /repos/:o/:r/pulls`;
  body = task + funnel table + models + cost + receipt JSON in a details block,
  footer "Generated by kolm code". Returns `{url, number}`. Env-gated by config (section 2); unarmed -> `LadderError('not-armed')`.
  `fetch` is injectable for tests.
* `truffle/partner-task.js`: implement the client side of
  `truffle.channel-partner-task.v1` exactly as defined in the Truffle repo
  (`C:\Users\user\Desktop\truffle-chat\src\lib\channel-workflow-partner\contract.ts`,
  `docs/channel-workflow-architecture.md`, migration 0118 harness catalog):
  claim -> lease/heartbeat with fence token -> settle (success with artifact
  = receipt + diff, or failure with reasons). Every request carries the
  fence; a stale fence aborts the run. Endpoints/base URL from config.
* `truffle/worker.js`: `runWorker({ config, deps, once })` polls for tasks
  whose harness id matches, maps the Truffle task payload to
  `runTask({task, repoPath|repoUrl, baseRef})` (cloning to a temp dir when
  only a URL is given), heartbeats during the run, settles with the receipt.
  Tests use an in-process fake Truffle HTTP server (`node:http`).

## 10. Verification plan (what "done" means)

1. `node --test --test-concurrency=1 tests/ladder-*.test.js` green on Windows.
2. End-to-end test (owner D, `tests/ladder-e2e.test.js`): temp git repo with a
   tiny Node project (`package.json` `test` script running `node --test`),
   one failing test; fake provider scripted so candidate 1 writes a wrong fix
   (L2 newly failing -> bad), repair writes the right fix (all good), fake
   judge calibration passes; asserts: run `passed`, winner branch exists in
   the repo with the commit, `results` rows show attempt ids and versions,
   receipt verifies, funnel numbers match, cost ledger sums, `resumeRun`
   on a half-finished run (simulate by aborting via signal after L1) completes without recomputing L0/L1 rows.
3. Local-git driver test against a real temp repo (skip if `git` missing).
4. `cd deploy/kolm-code-sandbox && npm test && npm run build` (dry-run deploy).
5. Existing kolm suite still green for files touched by the SDK bump:
   `node --test --test-concurrency=1 tests/*.test.js` (full).
6. Live smoke (run by the orchestrating session, not by owners):
   `KOLM_CODE_CODER_LADDER=ollama:qwen3.8:latest KOLM_CODE_REVIEWER_LADDER=ollama:qwen3.8:latest kolm code run "..." --repo <fixture>`
   with the real `ANTHROPIC_API_KEY` for planner/judge, and `kolm code doctor --live`.

## 11. Addendum from the repo maps (binding, supersedes earlier sections where they conflict)

### 11.1 kolm repo facts
* CI runs the full suite on **Node 20** (`.github/workflows/test-suite.yml`), local is Node 24.
  `node:sqlite` is only in Node >= 22.5. `src/event-store.js:187-196` shows the house pattern:
  `try { ({ DatabaseSync } = require('node:sqlite')) } catch { fallback }`.
  For `kolm code`: `openStore()` throws `LadderError('sqlite-unavailable')` with the Node
  version in the message when `node:sqlite` is missing (fail loud, no JSONL fallback),
  and **every ladder test file skips itself** (`test.skip` / `{ skip: reason }`) when
  `node:sqlite` cannot be required, so CI on Node 20 stays green. Export a tiny helper
  `sqliteAvailable()` from `src/ladder/store.js` for that.
* Testing policy (`docs/reference/testing-policy.md`): no new npm dependencies at all;
  tests must never touch the network (only `127.0.0.1` servers you spawn yourself); test
  labels are descriptive sentences (>= 6 chars) or `test_<topic>_<aspect>`; the word
  "honest*" is grepped for and forbidden; three sequential runs must produce identical
  pass/fail/skip counts (deterministic, no timing-based assertions).
* `KOLM_TEST_MOCK_PROVIDER=1` is the house switch for short-circuiting provider adapters.
  `createProvider()` must honor it: when set, return a deterministic fake provider
  (canned responses, `armed:true`, cost 0) instead of any network adapter.
* ESLint (`eslint.config.js`, flat, correctness rules only: `no-undef`, `no-unused-vars`
  with `argsIgnorePattern:'^_'`, `eqeqeq`, `no-var`, `no-unreachable`, `no-dupe-keys`,
  `no-cond-assign`, `no-constant-condition` with `checkLoops:false`) runs over
  `src cli scripts tests workers`. Run `npx eslint src/ladder cli/kolm-code.mjs tests/ladder-*.test.js`
  before you finish; zero errors.
* Reusable existing code you MAY import (read-only, do not edit):
  `src/ed25519.js` (signing), `src/providers/_shared.js` (`hardenedFetch` with
  Retry-After, backoff `[500,1500,4500]`, timeouts; optional), `src/cid.js`
  (`canonicalJson`, verify its shape before relying on it). Do not import
  `src/llm-call.js` (single global provider from env; wrong shape for a multi-provider cascade).
* The CLI doc-mirror tests (`tests/wave353-agent-guide.test.js`) only require that every
  verb in `src/intent.js` `VERB_DESCRIPTIONS` has a `case` in `cli/kolm.js`, not the reverse.
  So `case 'code'` + `COMPLETION_VERBS` + one help line is the complete wiring. Do not
  touch `src/intent.js` or `AGENT_GUIDE.md`.
* Cerebras key aliases already used elsewhere in kolm (`src/teachers/cerebras.js:51-53`):
  accept `CEREBRAS_API_KEY || KOLM_CEREBRAS_TOKEN`, base URL `KOLM_CEREBRAS_URL || https://api.cerebras.ai/v1`.
* Never use `innerHTML` with template strings in the viewer (a lint job bans it under
  `public/`; keep the same discipline in `src/ladder/viewer/index.html`): build DOM with
  `createElement` / `textContent`.

### 11.2 Truffle facts (owner F and owner B)
* The Truffle GitHub App key on this machine is the file
  `~/.truffle/truffle-pr-authority.private-key.pem` (exists, RSA PEM). Truffle itself
  consumes the key as the inline-PEM env var `TRUFFLE_GITHUB_PRIVATE_KEY` and the ids as
  `TRUFFLE_GITHUB_APP_ID` / `TRUFFLE_GITHUB_INSTALLATION_ID`. Config resolution for kolm:
  app id = `KOLM_GITHUB_APP_ID || TRUFFLE_GITHUB_APP_ID`; installation id =
  `KOLM_GITHUB_APP_INSTALLATION_ID || TRUFFLE_GITHUB_INSTALLATION_ID`; private key =
  `KOLM_GITHUB_APP_PRIVATE_KEY` (inline PEM) || `TRUFFLE_GITHUB_PRIVATE_KEY` (inline PEM)
  || file at `KOLM_GITHUB_APP_PRIVATE_KEY_PATH` || the `~/.truffle/...pem` file if present.
* Truffle's App is a read + check-run + merge gateway with **no** `contents:write` /
  `pull_requests:write`, and Truffle policy forbids expanding it. So the PR opener must
  also support a plain token: resolution order for the push/PR token is
  `KOLM_GITHUB_TOKEN` -> `GITHUB_TOKEN` -> GitHub App installation token (when app id +
  installation id + key all resolve) -> `gh auth token` (only when the `gh` CLI is on PATH;
  the machine is logged in as a GitHub user). `armingReport` reports which source armed
  `github`. Token values are never logged or written to disk.
* Truffle's partner contract is `truffle.channel-partner-task.v1`, defined in
  `C:\Users\user\Desktop\truffle-chat\src\lib\channel-workflow-partner\contract.ts` (1068
  lines) with the reference worker at `...\channel-workflow-partner\reference-worker.ts`.
  Clone the reference worker's behavior, not a simplified version:
  - Gateway endpoints `POST /api/channel-workflow-partner/v1/claim` and `POST .../v1/settle`.
  - Auth headers: `Authorization: Bearer <43-char base64url token>`,
    `x-truffle-service-audience: truffle-collaboration`, `x-truffle-workspace-id: <uuid>`.
  - Readiness attestation `truffle.partner-processor-readiness.v1` (`observedAt` within
    -30 s / +5 s of now, `leaseTtlSeconds: 60`, `releaseSha` 40-hex, `processorEndpointDigest`
    hex64, `partnerAdapterKey: 'truffle.channel-partner-task.v1'`), re-proved with a signed
    `OPTIONS` probe before each claim poll.
  - HMAC-SHA256 signature `v1=<hex64>` over
    `truffle:partner-processor-request:v1\nPOST\n<target>\n<timestamp>\n<keyId>\n<idempotencyKey>\n<bodyDigest>`
    (response variant `truffle:partner-processor-response:v1\n<status>\n...`), max age 300 s;
    signed headers `content-type, idempotency-key, x-truffle-partner-contract,
    x-truffle-partner-content-sha256, x-truffle-partner-key-id,
    x-truffle-partner-replay-contract, x-truffle-partner-timestamp, x-truffle-partner-signature`.
  - Claim `{schema, runId, version, attempt, idempotencyKey(hex64), triggerText (<= 32000 chars),
    threadRootId, task{...instructions <= 4000 chars, stepIndex 0..7, output{kind:'thread-message', maxBytes <= 32000}}, leaseToken(hex64)}`;
    lease TTL 120 s.
  - Settlement `{schema, run{id,version,leaseToken}, capability(hex64), operationId: 'partner.settle.'+idempotencyKey, output{kind:'thread-message', text}}`;
    result `{schema, runId, status:'succeeded'|'rejected', messageId|null, replayed}`.
  - The **only** artifact Truffle accepts is one plain-text thread message (<= 32000 bytes):
    render the receipt summary (verdict, winner branch, funnel, cost, receipt sha256 + key
    fingerprint) as text; never attachments.
  - Env names (Truffle's own, reuse them; `KOLM_TRUFFLE_*` from section 2 are aliases):
    `TRUFFLE_PARTNER_API_URL`, `TRUFFLE_PARTNER_PROCESSOR_URL`, `TRUFFLE_PARTNER_SERVICE_TOKEN`,
    `TRUFFLE_PARTNER_WORKSPACE_ID`, `TRUFFLE_PARTNER_PROCESSOR_KEY_ID`,
    `TRUFFLE_PARTNER_PROCESSOR_SIGNING_SECRET`, `TRUFFLE_PARTNER_HEALTH_HOST` (0.0.0.0),
    `TRUFFLE_PARTNER_POLL_MS` (1000). Read `contract.ts` and `reference-worker.ts` in full
    before writing `truffle/partner-task.js`; copy constants (poll 1000 ms, request timeout
    10 s, processor timeout 20 s, 3 attempts, backoff 250 ms base, 2000 ms cap, 48000-byte
    max gateway response, 2048 max URL length).
  - Truffle's partner gateway is not production-activated (migration 0118 has no activation
    flag). The adapter is env-gated and verified only against the in-process fake server.
* Truffle's agent-home sandbox image (`cloudflare/sandbox:0.12.9-python`) ships **without
  git** and receives repos as tarballs. kolm's Cloudflare worker must therefore install git
  in its Dockerfile defensively:
  `RUN command -v git >/dev/null 2>&1 || (apt-get update && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*)`.
  Also allow the sandbox to reach `github.com` (clone) plus package registries: if the
  `Sandbox` subclass uses `enableInternet=false; interceptHttps=true; allowedHosts=[...]`
  (Truffle's `team-sandbox.mjs` pattern), the allowlist must include `github.com`,
  `codeload.github.com`, `objects.githubusercontent.com`, `registry.npmjs.org`, `pypi.org`,
  `files.pythonhosted.org`; make the list extensible via a `KOLM_CODE_SANDBOX_ALLOWED_HOSTS`
  var (comma-separated) in `wrangler.jsonc` vars.
* Cloudflare account id (for README only): `5d6e3a656784a798c0d763c9eed5214d`.
