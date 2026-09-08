// kolm code sandbox — the Durable Object container class (spec sections 8, 11.2).
//
// Network posture mirrors Truffle's TeamSandbox: egress is off by default and
// only re-opened for the source/package hosts a build actually needs, with TLS
// interception so the allowlist is enforced on the CONNECT target rather than
// on DNS alone. Extra hosts can be added per deployment through the
// `KOLM_CODE_SANDBOX_ALLOWED_HOSTS` var (comma separated); the list itself
// lives in `hosts.mjs` so it can be tested without the runtime SDK.

import { Sandbox } from '@cloudflare/sandbox';
import { DEFAULT_ALLOWED_HOSTS, resolveAllowedHosts } from './hosts.mjs';

export { DEFAULT_ALLOWED_HOSTS, parseAllowedHosts, resolveAllowedHosts } from './hosts.mjs';

export class KolmCodeSandbox extends Sandbox {
  enableInternet = false;
  interceptHttps = true;
  allowedHosts = [...DEFAULT_ALLOWED_HOSTS];
  sleepAfter = '20m';

  constructor(ctx, env) {
    super(ctx, env);
    // Class field initialisers have already run at this point, so this
    // assignment is the effective one.
    this.allowedHosts = resolveAllowedHosts(env?.KOLM_CODE_SANDBOX_ALLOWED_HOSTS);
  }

  createClientForTransport(transport) {
    const client = super.createClientForTransport(transport);
    if (transport === 'rpc' && client?.connOptions?.stub === this) {
      // The pinned SDK routes its private control handshake through
      // `stub.fetch`. Hand that client the inherited handler explicitly so the
      // public `fetch` below can stay closed.
      client.connOptions.stub = { fetch: (request) => super.fetch(request) };
    }
    return client;
  }

  // The container exposes no HTTP surface of its own; every call arrives
  // through the worker's authenticated API and reaches the container over RPC.
  fetch() {
    return new Response('Not found', { status: 404 });
  }
}

export default KolmCodeSandbox;
