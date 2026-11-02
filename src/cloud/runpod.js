// W910-D - One-click RunPod cloud deploy.
//
// Higher-level surface above src/cloud-providers/runpod.js. The lower-level
// module owns GraphQL traffic + serverless polling; this module owns the
// product-level operations that the /account/artifacts UI + /v1/deploy/runpod
// routes invoke:
//
//   * provisionPod(opts)   -> create a serving pod from a .kolm artifact
//   * tearDownPod(podId)   -> stop + delete the pod
//   * getPodStatus(podId)  -> {status, ready, openai_url, ...}
//   * getPodLogs(podId)    -> tail of recent log lines
//
// All routes return structured error envelopes (never raw stack traces). The
// settings/integrations page is the configured fall-back location for the
// API key, and missing-key errors include a deep link to it.
//
// Caveats / Constraints / Limitations:
//   1. We do NOT bake user weights into a custom container image here.
//      Instead the worker image (kolm/serve:latest) loads the artifact at
//      cold-start from KOLM_MODEL_URL. The caller is responsible for ensuring
//      that URL is reachable from the RunPod worker network (signed S3 URL
//      or kolm.ai registry URL).
//   2. RunPod's `podFindAndDeployOnDemand` mutation shape has changed twice
//      in the last 18 months. We default to the current shape but accept
//      `opts.mutations` for forward-compat - see src/cloud-providers/runpod.js
//      for the same override pattern.
//   3. Cost numbers in the cost-confirm UI are informational only - RunPod
//      bills directly via their console. We never hold a credit card on
//      behalf of the user for GPU time.

import { RunPodProvider, RUNPOD_GPU_CATALOG } from '../cloud-providers/runpod.js';

const POD_GRAPHQL_URL = 'https://api.runpod.io/graphql';
const DOCS_URL = 'https://docs.runpod.io/pods/manage-pods';

// Approximate per-hour pricing as of 2026-05. Used only for the cost-confirm
// modal - real billing comes from RunPod directly. Values are deliberately
// rounded up so we never under-quote the user.
export const RUNPOD_GPU_PRICING_USD_HR = Object.freeze({
  'RTX A4000':           0.30,
  'NVIDIA RTX A4000':    0.30,
  'RTX-A4000':           0.30,
  'A4000':               0.30,
  'RTX 4090':            0.69,
  'NVIDIA GeForce RTX 4090': 0.69,
  'RTX-4090':            0.69,
  'RTX 5090':            0.79,
  'RTX-5090':            0.79,
  'L40S':                1.19,
  'NVIDIA L40S':         1.19,
  'A100 40GB':           1.49,
  'A100-40GB':           1.49,
  'A100':                1.89,
  'A100 80GB':           1.89,
  'A100-80GB':           1.89,
  'NVIDIA A100 80GB PCIe': 1.89,
  'H100':                3.89,
  'H100-80GB':           3.89,
  'NVIDIA H100 80GB HBM3': 3.89,
});

export function priceForGpu(gpuLabel) {
  const key = String(gpuLabel || '').trim();
  if (!key) return null;
  if (Object.prototype.hasOwnProperty.call(RUNPOD_GPU_PRICING_USD_HR, key)) {
    return RUNPOD_GPU_PRICING_USD_HR[key];
  }
  const normalized = RUNPOD_GPU_CATALOG[key];
  if (normalized && Object.prototype.hasOwnProperty.call(RUNPOD_GPU_PRICING_USD_HR, normalized)) {
    return RUNPOD_GPU_PRICING_USD_HR[normalized];
  }
  return null;
}

// Pod-specific GraphQL mutations. These shape-match RunPod's "GPU Pods" API
// (the persistent-pod surface; distinct from serverless endpoints).
function _podMutations() {
  return {
    // POST a new pod. The `gpuTypeId` is the human-readable label
    // (e.g. "NVIDIA RTX A4000") not the numeric id - RunPod's resolver
    // accepts both.
    deployPod: {
      query: `mutation DeployPod($input: PodFindAndDeployOnDemandInput!) {
        podFindAndDeployOnDemand(input: $input) {
          id
          imageName
          machineId
          desiredStatus
          costPerHr
          gpuCount
          memoryInGb
          vcpuCount
        }
      }`,
    },
    stopPod: {
      query: `mutation StopPod($input: PodStopInput!) {
        podStop(input: $input) { id desiredStatus }
      }`,
    },
    terminatePod: {
      query: `mutation TerminatePod($input: PodTerminateInput!) {
        podTerminate(input: $input)
      }`,
    },
    getPod: {
      query: `query GetPod($input: PodFilter!) {
        pod(input: $input) {
          id
          name
          imageName
          desiredStatus
          lastStatusChange
          costPerHr
          runtime {
            uptimeInSeconds
            ports { ip isIpPublic privatePort publicPort type }
            gpus { id gpuUtilPercent memoryUtilPercent }
            container { cpuPercent memoryPercent }
          }
          machine { gpuTypeId podHostId }
        }
      }`,
    },
    getPodLogs: {
      query: `query PodLogs($podId: String!) {
        pod(input: { podId: $podId }) {
          id
          logs
        }
      }`,
    },
  };
}

// _callGraphQL - thin wrapper around fetch. Returns parsed `data` or throws
// a tagged error. Identical contract to cloud-providers/runpod.js so callers
// can treat error envelopes interchangeably.
async function _callGraphQL({ apiKey, query, variables, opName, fetchImpl, proxy }) {
  const fetchFn = fetchImpl || globalThis.fetch;
  if (typeof fetchFn !== 'function') {
    const err = new Error('global fetch is required (Node 22+). Pass opts.fetch to override.');
    err.code = 'fetch_unavailable';
    throw err;
  }
  // No direct key? Route through the Vercel proxy (api/runpod.js), which holds
  // the operator's runpod_api_key. The proxy returns the same { data, errors }
  // GraphQL shape as api.runpod.io, so all parsing below is unchanged.
  const px = proxy || (!apiKey ? _resolveProxy() : null);
  const useProxy = !apiKey && px && px.url;
  let res;
  try {
    res = await fetchFn(useProxy ? px.url : POD_GRAPHQL_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${useProxy ? px.bearer : apiKey}`,
      },
      body: JSON.stringify({ query, variables }),
    });
  } catch (e) {
    const err = new Error(`runpod ${opName} fetch failed: ${e.message}`);
    err.code = 'runpod_fetch_failed';
    err.op = opName;
    err.docs_url = DOCS_URL;
    throw err;
  }
  const text = await res.text().catch(() => '');
  if (!res.ok) {
    const err = new Error(`runpod ${opName} http ${res.status}: ${text.slice(0, 500)}`);
    err.code = 'runpod_http_error';
    err.status = res.status;
    err.op = opName;
    err.body = text.slice(0, 2000);
    err.docs_url = DOCS_URL;
    throw err;
  }
  let json;
  try { json = JSON.parse(text); }
  catch (e) {
    const err = new Error(`runpod ${opName} parse error: ${e.message}`);
    err.code = 'runpod_parse_error';
    err.op = opName;
    err.body = text.slice(0, 2000);
    throw err;
  }
  if (Array.isArray(json.errors) && json.errors.length) {
    const msg = json.errors.map((g) => g.message || JSON.stringify(g)).join('; ');
    const err = new Error(`runpod ${opName} graphql error: ${msg}`);
    err.code = 'runpod_graphql_error';
    err.op = opName;
    err.errors = json.errors;
    err.docs_url = DOCS_URL;
    // Surface specific actionable errors for "no GPU available".
    if (/no\s*gpu|capacity|out\s*of\s*stock/i.test(msg)) {
      err.code = 'runpod_no_gpu_available';
      err.alternatives = _alternativeGpus(variables && variables.input && variables.input.gpuTypeId);
    }
    throw err;
  }
  return json.data || {};
}

function _alternativeGpus(currentLabel) {
  const order = ['RTX A4000', 'RTX 4090', 'L40S', 'A100', 'A100 80GB', 'H100'];
  const cur = String(currentLabel || '').toLowerCase();
  return order.filter((g) => !cur.includes(g.toLowerCase().split(' ')[0])).slice(0, 4);
}

function _resolveApiKey(opts = {}) {
  return (opts && opts.apiKey)
    || process.env.RUNPOD_API_KEY
    || process.env.KOLM_RUNPOD_TOKEN
    // Operators commonly add the key with Vercel's lowercase casing; read it too
    // (server.js normalizeEnv() also maps this, but be robust in any process).
    || process.env.runpod_api_key
    || '';
}

// When there is NO direct RunPod key in this process (the common case: the
// operator keeps runpod_api_key in Vercel, which the API router + CLI can't
// see), fall back to the Vercel proxy (api/runpod.js) which DOES hold the key.
// Requires a kolm bearer + base to authenticate to the proxy.
function _resolveProxy(opts = {}) {
  const bearer = (opts && (opts.proxyBearer || opts.kolmKey))
    || process.env.KOLM_API_KEY || process.env.KOLM_KEY || '';
  if (!bearer) return null;
  const base = String((opts && opts.proxyBase) || process.env.KOLM_BASE_URL
    || process.env.KOLM_BASE || 'https://kolm.ai').replace(/\/+$/, '');
  return { url: base + '/v1/runpod/graphql', bearer };
}

// Resolve either a direct key or the Vercel proxy; throw the actionable
// missing-key error only when NEITHER is available.
function _resolveAuth(opts = {}) {
  const apiKey = _resolveApiKey(opts);
  const proxy = apiKey ? null : _resolveProxy(opts);
  if (!apiKey && !proxy) throw _missingKeyError();
  return { apiKey, proxy };
}

function _missingKeyError() {
  const err = new Error('RunPod API key missing. Configure at /account/settings/integrations or export RUNPOD_API_KEY.');
  err.code = 'runpod_api_key_missing';
  err.install_hint = 'Open /account/settings/integrations and paste a RunPod API key (created at https://runpod.io/console/user/settings).';
  err.docs_url = 'https://docs.runpod.io/get-started/api-keys';
  err.action_url = '/account/settings/integrations';
  return err;
}

// provisionPod({ apiKey, gpu_type, container_image, env, artifact_url, name, fetch })
//   -> { ok, pod_id, openai_url_pending, status, cost_per_hr, gpu_type, raw }
//
// The container is expected to expose an OpenAI-compatible /v1/chat/completions
// on port 8000 once warm. The OpenAI URL is reported as pending until the pod
// transitions to RUNNING and `runtime.ports` shows a public port - callers
// should poll getPodStatus() until openai_url is set.
export async function provisionPod(opts = {}) {
  const apiKey = _resolveApiKey(opts);
  if (!apiKey && !_resolveProxy(opts)) throw _missingKeyError();
  const gpuLabel = opts.gpu_type || opts.gpuType || 'RTX A4000';
  const gpuTypeId = RUNPOD_GPU_CATALOG[gpuLabel] || gpuLabel;
  const containerImage = opts.container_image || opts.containerImage || 'kolm/serve:latest';
  const artifactUrl = opts.artifact_url || opts.artifactUrl || (opts.env && (opts.env.KOLM_MODEL_URL || opts.env.KOLM_ARTIFACT_URL)) || '';
  const baseEnv = {
    KOLM_MODEL_URL: artifactUrl,
    KOLM_ARTIFACT_URL: artifactUrl,
    KOLM_PORT: '8000',
    ...(opts.env || {}),
  };
  const envList = Object.entries(baseEnv).map(([key, value]) => ({ key, value: String(value == null ? '' : value) }));
  const name = opts.name || `kolm-serve-${Date.now().toString(36)}`;
  const variables = {
    input: {
      cloudType: opts.cloud_type || 'COMMUNITY',
      gpuCount: opts.gpu_count || 1,
      volumeInGb: opts.volume_gb != null ? opts.volume_gb : 50,
      containerDiskInGb: opts.container_disk_gb != null ? opts.container_disk_gb : 20,
      minVcpuCount: opts.min_vcpu || 2,
      minMemoryInGb: opts.min_ram_gb || 8,
      gpuTypeId,
      name,
      imageName: containerImage,
      dockerArgs: opts.docker_args || '',
      ports: opts.ports || '8000/http',
      volumeMountPath: opts.volume_mount_path || '/workspace',
      env: envList,
    },
  };
  const data = await _callGraphQL({
    apiKey,
    query: _podMutations().deployPod.query,
    variables,
    opName: 'deployPod',
    fetchImpl: opts.fetch,
  });
  const pod = data.podFindAndDeployOnDemand || {};
  const podId = pod.id || null;
  if (!podId) {
    const err = new Error('runpod deployPod returned no pod id (RunPod likely rejected the request silently).');
    err.code = 'runpod_no_pod_id';
    err.raw = pod;
    throw err;
  }
  return {
    ok: true,
    provider: 'runpod',
    pod_id: podId,
    status: pod.desiredStatus || 'PROVISIONING',
    cost_per_hr: typeof pod.costPerHr === 'number' ? pod.costPerHr : (priceForGpu(gpuLabel) || null),
    gpu_type: gpuLabel,
    gpu_count: pod.gpuCount || 1,
    image: pod.imageName || containerImage,
    openai_url_pending: true,
    artifact_url: artifactUrl || null,
    docs_url: DOCS_URL,
    raw: pod,
  };
}

// tearDownPod(podId, { apiKey, fetch })
//   Best effort: issues stopPod first, then terminatePod. Returns {ok, stopped, terminated, errors}
export async function tearDownPod(podId, opts = {}) {
  if (!podId) {
    const err = new Error('tearDownPod requires podId'); err.code = 'bad_args'; throw err;
  }
  const apiKey = _resolveApiKey(opts);
  if (!apiKey && !_resolveProxy(opts)) throw _missingKeyError();
  const errors = [];
  let stopped = false, terminated = false;
  try {
    await _callGraphQL({
      apiKey,
      query: _podMutations().stopPod.query,
      variables: { input: { podId } },
      opName: 'stopPod',
      fetchImpl: opts.fetch,
    });
    stopped = true;
  } catch (e) {
    errors.push({ op: 'stopPod', code: e.code || 'unknown', message: e.message });
  }
  try {
    await _callGraphQL({
      apiKey,
      query: _podMutations().terminatePod.query,
      variables: { input: { podId } },
      opName: 'terminatePod',
      fetchImpl: opts.fetch,
    });
    terminated = true;
  } catch (e) {
    errors.push({ op: 'terminatePod', code: e.code || 'unknown', message: e.message });
  }
  return {
    ok: terminated, // primary success is termination; stop is best-effort first
    provider: 'runpod',
    pod_id: podId,
    stopped,
    terminated,
    errors: errors.length ? errors : null,
  };
}

// getPodStatus(podId, opts) -> { ok, pod_id, status, ready, openai_url, runtime, cost_per_hr }
export async function getPodStatus(podId, opts = {}) {
  if (!podId) {
    const err = new Error('getPodStatus requires podId'); err.code = 'bad_args'; throw err;
  }
  const apiKey = _resolveApiKey(opts);
  if (!apiKey && !_resolveProxy(opts)) throw _missingKeyError();
  const data = await _callGraphQL({
    apiKey,
    query: _podMutations().getPod.query,
    variables: { input: { podId } },
    opName: 'getPod',
    fetchImpl: opts.fetch,
  });
  const pod = data.pod || {};
  const runtime = pod.runtime || {};
  const ports = Array.isArray(runtime.ports) ? runtime.ports : [];
  // Locate the public HTTP port mapped from container 8000 - that's the
  // OpenAI-compatible endpoint.
  const publicPort = ports.find((p) => p && p.isIpPublic && (p.privatePort === 8000 || p.publicPort)) || null;
  const openaiUrl = publicPort ? `https://${pod.id}-${publicPort.publicPort}.proxy.runpod.net/v1` : null;
  const status = pod.desiredStatus || 'UNKNOWN';
  const ready = status === 'RUNNING' && !!openaiUrl;
  return {
    ok: true,
    provider: 'runpod',
    pod_id: pod.id || podId,
    name: pod.name || null,
    status,
    ready,
    openai_url: openaiUrl,
    uptime_seconds: runtime.uptimeInSeconds || 0,
    cost_per_hr: typeof pod.costPerHr === 'number' ? pod.costPerHr : null,
    gpu_type: pod.machine && pod.machine.gpuTypeId || null,
    image: pod.imageName || null,
    runtime: {
      gpus: runtime.gpus || [],
      container: runtime.container || null,
      ports,
    },
    raw: pod,
  };
}

// getPodLogs(podId, { tail, apiKey, fetch }) -> { ok, pod_id, lines: [string], total_lines }
export async function getPodLogs(podId, opts = {}) {
  if (!podId) {
    const err = new Error('getPodLogs requires podId'); err.code = 'bad_args'; throw err;
  }
  const apiKey = _resolveApiKey(opts);
  if (!apiKey && !_resolveProxy(opts)) throw _missingKeyError();
  const tail = Math.max(1, Math.min(2000, Number(opts.tail) || 200));
  const data = await _callGraphQL({
    apiKey,
    query: _podMutations().getPodLogs.query,
    variables: { podId },
    opName: 'getPodLogs',
    fetchImpl: opts.fetch,
  });
  const pod = data.pod || {};
  const raw = String(pod.logs || '');
  const lines = raw ? raw.split(/\r?\n/) : [];
  return {
    ok: true,
    provider: 'runpod',
    pod_id: podId,
    lines: lines.slice(-tail),
    total_lines: lines.length,
    truncated: lines.length > tail,
  };
}

// testConnection(opts) -> { ok, reason?, account?, gpu_count_available? }
//   Light-weight reachability + credential check for the "Test connection"
//   button on /account/settings/integrations. Never throws.
export async function testConnection(opts = {}) {
  const apiKey = _resolveApiKey(opts);
  if (!apiKey) {
    return {
      ok: false,
      reason: 'no_api_key',
      install_hint: 'Paste a RunPod API key from https://runpod.io/console/user/settings.',
      action_url: '/account/settings/integrations',
    };
  }
  try {
    const data = await _callGraphQL({
      apiKey,
      query: `query Me { myself { id email currentSpendPerHr } }`,
      variables: {},
      opName: 'testConnection',
      fetchImpl: opts.fetch,
    });
    const me = (data && data.myself) || null;
    return {
      ok: !!me,
      provider: 'runpod',
      account: me ? { id: me.id, email: me.email, current_spend_per_hr: me.currentSpendPerHr } : null,
    };
  } catch (e) {
    return {
      ok: false,
      provider: 'runpod',
      reason: e.code || 'unknown',
      message: e.message,
      docs_url: e.docs_url || DOCS_URL,
    };
  }
}

// Re-export the serverless provider so callers that need both the persistent
// pod surface (this file) and the serverless endpoint surface (the parent
// module) can resolve everything off one import.
export { RunPodProvider, RUNPOD_GPU_CATALOG };

export default {
  provisionPod,
  tearDownPod,
  getPodStatus,
  getPodLogs,
  testConnection,
  priceForGpu,
  RUNPOD_GPU_PRICING_USD_HR,
  RUNPOD_GPU_CATALOG,
};
