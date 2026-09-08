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
} from './config.js';

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

// --- evaluation -------------------------------------------------------------
export { evalRuns, formatEvalTable } from './eval.js';
