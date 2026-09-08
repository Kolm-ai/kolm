// kolm code — worktree driver factory (owner B, spec section 4).
//
//   const driver = createDriver(config.driver, config, opts);
//
// Every driver exposes the same surface:
//   driver.kind, driver.create(...), driver.open(...), driver.destroyRun(runId)
// and every worktree exposes:
//   id, kind, path, exec, readFile, writeFile, deleteFile, listFiles, search,
//   diff, changedFiles, applyPatch, commit, baseSha, destroy

import { LadderError } from '../errors.js';
import { createLocalGitDriver } from './local-git.js';
import { createCloudflareSandboxDriver } from './cloudflare-sandbox.js';
import { createFakeDriver } from './fake.js';

export const DRIVER_KINDS = Object.freeze(['local', 'cloudflare', 'fake']);

/**
 * @param {'local'|'cloudflare'|'fake'} kind
 * @param {object} [config] ladder config (worktreesDir / sandboxUrl /
 *                          sandboxToken / stageTimeoutMs are the only fields read)
 * @param {object} [opts]   per-call overrides, passed through to the driver
 */
export function createDriver(kind, config = {}, opts = {}) {
  switch (kind) {
    case 'local':
      return createLocalGitDriver(config, opts);
    case 'cloudflare':
      return createCloudflareSandboxDriver(config, opts);
    case 'fake':
      return createFakeDriver({ ...opts });
    default:
      throw new LadderError(
        'bad-config',
        `unknown worktree driver ${JSON.stringify(kind)} (expected one of ${DRIVER_KINDS.join(', ')})`,
        { kind, supported: [...DRIVER_KINDS] },
      );
  }
}

export { createLocalGitDriver } from './local-git.js';
export { createCloudflareSandboxDriver } from './cloudflare-sandbox.js';
export { createFakeDriver } from './fake.js';

export default createDriver;
