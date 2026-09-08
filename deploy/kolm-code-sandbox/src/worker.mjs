// kolm code sandbox — worker entrypoint (spec section 8).
//
// One container per run id: `getSandbox(env.SANDBOX, runId)` gives every
// worktree of a run the same filesystem, and `DELETE /v1/runs/:runId` tears it
// down. All routing, path safety and git orchestration live in `api.mjs`.

import { getSandbox } from '@cloudflare/sandbox';
import { KolmCodeSandbox } from './sandbox.mjs';
import { createApi } from './api.mjs';

export { KolmCodeSandbox };

export default {
  async fetch(request, env) {
    const api = createApi({
      getSandbox: (id) => getSandbox(env.SANDBOX, id),
      token: env.KOLM_CODE_SANDBOX_TOKEN,
    });
    return api.handle(request);
  },
};
