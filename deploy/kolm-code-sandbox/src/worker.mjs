// kolm code sandbox — worker entrypoint (spec section 8).
//
// One container per run id: `getSandbox(env.SANDBOX, runId)` gives every
// worktree of a run the same filesystem, and `DELETE /v1/runs/:runId` tears it
// down. All routing, path safety and git orchestration live in `api.mjs`.

import { ContainerProxy, getSandbox } from '@cloudflare/sandbox';
import { KolmCodeSandbox } from './sandbox.mjs';
import { createApi } from './api.mjs';

export { KolmCodeSandbox };
// The pinned SDK starts the container through `ctx.exports.ContainerProxy`;
// without this export every start fails with "ctx.exports.ContainerProxy is
// undefined" and the container never becomes ready (seen live 2026-09-10).
export { ContainerProxy };

export default {
  async fetch(request, env) {
    const api = createApi({
      getSandbox: (id) => getSandbox(env.SANDBOX, id),
      token: env.KOLM_CODE_SANDBOX_TOKEN,
    });
    return api.handle(request);
  },
};
