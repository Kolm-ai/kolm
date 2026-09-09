// kolm code — the single public surface (spec 6.7). Owner D.
//
// Everything outside src/ladder/ (the CLI, the viewer, the GitHub and Truffle
// helpers, other repos) imports from HERE and nowhere else. That keeps the
// internal file layout free to move: the module graph below is the contract.
//
// Load order note: this file is written first on purpose, so the CLI owner can
// integrate against it while the orchestrator internals are still landing.

export const VERSION = 'kolm-code/0.1.0';

// --- errors -----------------------------------------------------------------
export { LadderError, isLadderError, LADDER_ERROR_CODES } from './errors.js';

// --- config / arming --------------------------------------------------------
export {
  loadConfig,
  armingReport,
  effectiveLadders,
  probeOllama,
  probeGhCli,
  parseRef,
  expandHome,
  withOllamaReachable,
  DEFAULT_ORCHESTRATOR,
  DEFAULT_CODER_LADDER,
  DEFAULT_REVIEWER_LADDER,
  DEFAULT_RETRIEVAL_BYTES,
  FILE_SECRET_NAMES,
  ROUTING_MODES,
} from './config.js';

// --- keystore (bring your own key) -----------------------------------------
export {
  keystorePath,
  keystoreDisabled,
  readKeystore,
  setKey,
  unsetKey,
  describeKeys,
  assertKeyName,
  fingerprint,
  probeKey,
  KEY_NAMES,
  KEY_ENV_NAMES,
  KEYSTORE_VERSION,
} from './keystore.js';

// --- providers --------------------------------------------------------------
export {
  createProvider,
  describeModel,
  resolveRef,
  createFakeProvider,
  createMockProvider,
  estimateCost,
  listModels,
  knownRefs,
  SUPPORTED_PROVIDERS,
  TEXT_TOOL_PROTOCOL_VERSION,
  parseTextToolCalls,
  renderToolsForText,
  renderToolResultsForText,
  toTextProtocolMessages,
  renderJsonOnlyForText,
  refForcedToTextTools,
  looksLikeToolsUnsupported,
  PROVIDER_CONCURRENCY_DEFAULTS,
  THROTTLE_PENALTY_MS,
  createLimiter,
  limiterFor,
  resetLimiters,
  retryDelayMs,
} from './providers/index.js';

// --- worktree drivers -------------------------------------------------------
export { createDriver, DRIVER_KINDS, createFakeDriver } from './worktree/index.js';

// --- store ------------------------------------------------------------------
export { openStore, sqliteAvailable, newRunId, candidateIdFor, SCHEMA_VERSION } from './store.js';

// --- bands / rubric / verdict ----------------------------------------------
export { loadBands, band, bandFor, worst, validateBands, BUNDLED_BANDS_PATH } from './bands.js';
export {
  loadRubric,
  rubricToPrompt,
  scoreRubric,
  blockingFailures,
  BLOCKING_ITEM_IDS,
  BUNDLED_RUBRIC_PATH,
} from './rubric.js';
export { assembleVerdict, flowsUp, funnel } from './verdict.js';

// --- context ----------------------------------------------------------------
export { detectCommands, parseTestOutput } from './context/detect.js';
export { repoBrief, briefToPrompt } from './context/repo-brief.js';
export { baseline, runCommand } from './context/baseline.js';
export {
  retrieveContext,
  retrievalToPrompt,
  extractTerms,
  searchPatternFor,
  toAscii,
  isVendored,
  posixPath,
  extOf,
  VENDOR_DIRS,
  DEFAULT_MAX_FILES,
  DEFAULT_MAX_BYTES,
} from './context/retrieve.js';
export {
  loadInstructions,
  instructionsToPrompt,
  INSTRUCTIONS_BYTE_CAP,
  INSTRUCTION_SOURCES,
} from './context/instructions.js';

// --- transcript compaction --------------------------------------------------
export { estimateTokens, compactTranscript } from './transcript.js';

// --- lessons (cross-run memory) ---------------------------------------------
export {
  LESSONS_VERSION,
  LESSONS_CAP,
  LESSONS_BYTE_CAP,
  LESSON_TEXT_CAP,
  LESSON_CANDIDATE_CAP,
  repoKeySlug,
  lessonsPath,
  loadLessons,
  saveLessons,
  mergeLessons,
  lessonsToPrompt,
  extractLessonCandidates,
} from './lessons.js';

// --- cascade ----------------------------------------------------------------
export { STAGES, runCascade } from './stages/index.js';

// --- judge (L4) -------------------------------------------------------------
export {
  runJudge,
  selectCalibrationSample,
  calibrationReport,
  suggestBands,
  JUDGMENT_SCHEMA,
} from './stages/l4-judge.js';

// --- planner / coder --------------------------------------------------------
export { plan, PLAN_SCHEMA } from './planner.js';
export { runCoder, runRepair, CODER_TOOLS, DEFAULT_ALLOWED_COMMANDS } from './coder.js';

// --- orchestrator -----------------------------------------------------------
export { runTask, resumeTask, rankCandidates, pickWinner } from './orchestrator.js';

// --- receipts ---------------------------------------------------------------
export {
  buildReceipt,
  signReceipt,
  verifyReceipt,
  writeReceipt,
  readReceipt,
  receiptPathFor,
  RECEIPT_SCHEMA,
  RECEIPT_VERSION,
} from './receipt.js';

// --- routing ----------------------------------------------------------------
// Owned by the orchestrator half of the build; re-exported here so the CLI and
// the bench never reach into src/ladder/routing.js directly.
export {
  routeRun,
  normalizeComplexity,
  COMPLEXITY_LEVELS,
  MAX_ROUTED_CANDIDATES,
} from './routing.js';

// --- repair coach + subtask split -------------------------------------------
export {
  coachRepair,
  transcriptTailFor,
  normalizeHints,
  COACH_SCHEMA,
} from './coach.js';
export {
  validateSubtasks,
  topoLevels,
  subtaskCandidateIdFor,
  parseSubtaskCandidateId,
  subPlanFor,
  scopedAcceptance,
  conflictFeedback,
  finishedSubtasks,
  priorSubtaskVerdicts,
} from './subtasks.js';

// --- evaluation -------------------------------------------------------------
export { evalRuns, formatEvalTable } from './eval.js';

// --- benchmark --------------------------------------------------------------
export {
  listBenchTasks,
  runBench,
  formatBenchTable,
  compareBench,
  probeRequirement,
  seedBenchRepo,
  hiddenCommandFor,
  benchCountersFor,
  BENCH_TASKS_DIR,
  BENCH_RESULT_VERSION,
  SKIP_VERDICT,
} from './bench/index.js';
