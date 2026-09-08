// kolm code — ladder error type (owner A, spec section 0).
//
// One error class for the whole ladder. Every failure that a caller can act on
// carries a stable machine-readable `code`; `detail` carries structured context
// (status codes, response bodies, refs) that must never be parsed out of the
// message string.

/**
 * Stable codes used across the ladder. Not an exhaustive allow-list — the
 * constructor accepts any string so a module can add a code without editing
 * this file — but every code that crosses a module boundary belongs here.
 */
export const LADDER_ERROR_CODES = Object.freeze([
  'not-armed',
  'provider-bad-request',
  'provider-unavailable',
  'provider-timeout',
  'budget-exceeded',
  'sqlite-unavailable',
  'bad-ref',
  'bad-config',
  // codes added by the store, stages, worktree, github and truffle modules
  'bad-arg',
  'bad-bands',
  'bad-rubric',
  'aborted',
  'store-closed',
  'path-escape',
  'execution',
  'judge-failed',
  'not-a-git-repo',
  'git-failed',
  'git-push',
  'worktree-exists',
  'worktree-missing',
  'sandbox-http',
  'github-app-auth',
  'github-pr',
  'truffle-config',
  'truffle-protocol',
  'truffle-gateway',
  'truffle-aborted',
  'truffle-processor-timeout',
]);

export class LadderError extends Error {
  /**
   * @param {string} code   stable machine code, e.g. 'not-armed'
   * @param {string} message human readable, safe to print (never a secret)
   * @param {object} [detail] structured context for programmatic handling
   */
  constructor(code, message, detail) {
    super(message);
    this.name = 'LadderError';
    this.code = code;
    this.detail = detail;
    if (Error.captureStackTrace) Error.captureStackTrace(this, LadderError);
  }

  toJSON() {
    return { name: this.name, code: this.code, message: this.message, detail: this.detail };
  }
}

/**
 * True when `e` is a LadderError. Uses a duck-typed fallback so errors that
 * cross a module realm (worker threads, dynamic imports) still classify.
 * @param {unknown} e
 * @param {string} [code] optional code to also match
 * @returns {boolean}
 */
export function isLadderError(e, code) {
  if (!e || typeof e !== 'object') return false;
  const ok = e instanceof LadderError || (e.name === 'LadderError' && typeof e.code === 'string');
  if (!ok) return false;
  return code === undefined ? true : e.code === code;
}
