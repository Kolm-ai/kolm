// Egress allowlist for the kolm code container (spec section 11.2).
//
// Kept in its own module because `src/sandbox.mjs` imports the Cloudflare
// runtime SDK, which only resolves inside a bundler; this file is plain Node so
// the allowlist stays unit-testable.

/** Hosts required to clone from GitHub and install public packages. */
export const DEFAULT_ALLOWED_HOSTS = Object.freeze([
  'github.com',
  'codeload.github.com',
  'objects.githubusercontent.com',
  'raw.githubusercontent.com',
  'api.github.com',
  'registry.npmjs.org',
  'pypi.org',
  'files.pythonhosted.org',
]);

/** Parse the `KOLM_CODE_SANDBOX_ALLOWED_HOSTS` var into a deduplicated list. */
export function parseAllowedHosts(value) {
  if (typeof value !== 'string' || value.trim() === '') return [];
  const out = [];
  for (const raw of value.split(',')) {
    const host = raw.trim().toLowerCase();
    if (host === '') continue;
    if (!/^[a-z0-9*][a-z0-9.*-]{0,253}$/.test(host)) continue;
    if (!out.includes(host)) out.push(host);
  }
  return out;
}

/** The effective allowlist for a deployment. */
export function resolveAllowedHosts(value) {
  const extra = parseAllowedHosts(value).filter((h) => !DEFAULT_ALLOWED_HOSTS.includes(h));
  return [...DEFAULT_ALLOWED_HOSTS, ...extra];
}
