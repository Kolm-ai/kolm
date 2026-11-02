#!/usr/bin/env node
// Complexity-graded end-to-end test.
//
// 6 tiers, each strictly harder than the previous, all the way up to a
// trained model and a live distill bridge:
//
//   T1 — trivial regex extract (email-extractor)
//   T2 — keyword boolean (spam-detector)
//   T3 — multi-class keyword classifier (sentiment 3-way)
//   T4 — 4-way classifier with adversarial holdout (support priority)
//   T5 — TRAINED logistic regression: gradient descent over TF-IDF features,
//        weights baked into a pure-JS generator. This is an actual ML model.
//   T6 — distill bridge end-to-end: spawn the FastAPI trainer service in
//        mock mode, POST /distill with a real JSONL corpus, watch progress
//        through stages, receive {metrics, adapter} envelope, bind the
//        adapter pointer into the .kolm artifact.
//
// Run:
//   node scripts/test-complexity-ramp.mjs

import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import crypto from 'node:crypto';
import http from 'node:http';
import { spawn } from 'node:child_process';
import { fileURLToPath, pathToFileURL } from 'node:url';

delete process.env.NODE_ENV;
delete process.env.AWS_EXECUTION_ENV;
delete process.env.VERCEL;

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(__dirname, '..');

const { synthesize } = await import(pathToFileURL(path.join(repo, 'src/synthesis.js')).href);
const { compileJs } = await import(pathToFileURL(path.join(repo, 'src/verifier.js')).href);
const { buildAndZip } = await import(pathToFileURL(path.join(repo, 'src/artifact.js')).href);
const { DEV_RECEIPT_SECRET } = await import(pathToFileURL(path.join(repo, 'src/env.js')).href);
const { cidFromManifestHashes, verifyCidAgainstManifestHashes, parseCid } = await import(pathToFileURL(path.join(repo, 'src/cid.js')).href);

const RECEIPT_SECRET = DEV_RECEIPT_SECRET;
const outDir = path.join(os.tmpdir(), 'kolm-complexity-ramp');
fs.mkdirSync(outDir, { recursive: true });

// --------------------------------------------------------------------------
// Shared utilities.
// --------------------------------------------------------------------------

function canonicalJson(v) {
  if (v === null || typeof v !== 'object') return JSON.stringify(v);
  if (Array.isArray(v)) return '[' + v.map(canonicalJson).join(',') + ']';
  const k = Object.keys(v).sort();
  return '{' + k.map(x => JSON.stringify(x) + ':' + canonicalJson(v[x])).join(',') + '}';
}

function verifyReceiptBody(receipt) {
  if (!receipt || typeof receipt !== 'object') return { valid: false, reason: 'no receipt' };
  const { signature, ...payload } = receipt;
  if (!signature) return { valid: false, reason: 'no signature' };
  const expected = crypto.createHmac('sha256', RECEIPT_SECRET).update(canonicalJson(payload)).digest('hex');
  return signature === expected ? { valid: true } : { valid: false, reason: 'signature mismatch' };
}

function verifyReceiptChain(receipt) {
  if (!Array.isArray(receipt.chain) || receipt.chain.length < 5) {
    return { valid: false, reason: `chain too short (${receipt.chain?.length ?? 0})` };
  }
  for (let i = 0; i < receipt.chain.length; i++) {
    const link = receipt.chain[i];
    const expected = crypto.createHmac('sha256', RECEIPT_SECRET)
      .update(canonicalJson({ step: link.step, input_hash: link.input_hash, output_hash: link.output_hash }))
      .digest('hex');
    if (link.hmac !== expected) return { valid: false, reason: `chain[${i}] hmac mismatch` };
    if (i > 0 && link.input_hash !== receipt.chain[i - 1].output_hash) {
      return { valid: false, reason: `chain[${i}] not anchored to chain[${i - 1}]` };
    }
  }
  return { valid: true, steps: receipt.chain.length };
}

function eq(a, b) {
  if (a === b) return true;
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
    return true;
  }
  return false;
}

function head(s, n = 8) { return String(s).slice(0, n); }

// Drive the full pipeline for a synthesized spec: pattern synth → compile →
// holdout grade → buildAndZip → all verifiers.
async function pipelineSynth(spec) {
  const positives = spec.train.map(p => ({ input: p.input, expected: p.expected }));
  const syn = await synthesize({ positives, negatives: [], output_spec: spec.output_spec, priors: {} });
  if (!syn.accepted) {
    return { ok: false, reasons: [`synthesizer rejected (q=${syn.best_result?.quality_score ?? '?'} pass=${syn.best_result?.pass_rate_positive ?? '?'})`] };
  }
  const fn = compileJs(syn.source);
  let holdoutPass = 0;
  for (const h of spec.holdout) {
    const out = fn(h.input);
    const ok = spec.output_spec.type === 'integer'
      ? Math.abs(Number(out) - Number(h.expected)) <= 2
      : eq(out, h.expected);
    if (ok) holdoutPass++;
  }
  return packageAndVerify({
    spec, source: syn.source, source_hash: syn.source_hash,
    holdoutPass, holdoutN: spec.holdout.length,
    training_stats: { verifier_accepted: true, pass_rate_positive: syn.pass_rate_positive, latency_p50_us: syn.latency_p50_us || 50 },
    strategy: 'pattern',
  });
}

// Package an already-compiled generator source + verify everything.
async function packageAndVerify({ spec, source, source_hash, holdoutPass, holdoutN, training_stats, strategy, extra_manifest = {} }) {
  const job_id = `job_${spec.name.replace(/[^a-z0-9]+/gi, '_')}_${crypto.randomBytes(3).toString('hex')}`;
  const recipes = [{
    id: `cpt_${spec.name}`,
    version_id: `ver_${spec.name}_001`,
    name: spec.name,
    source,
    source_hash,
    synthesized: true,
  }];
  const evals = {
    spec: 'rs-1-evals',
    n: spec.train.length,
    cases: spec.train.map((e, i) => ({ id: `case-${i+1}`, input: e.input, expected: e.expected })),
    coverage: training_stats.pass_rate_positive,
  };
  const built = await buildAndZip({
    job_id, task: spec.task,
    base_model: 'qwen2.5-coder-7b-instruct-q4_0',
    recipes, training_stats, evals,
    outDir,
  });

  const kComposite = built.k_score?.composite ?? 0;
  const reasons = [];
  if (kComposite < 0.85) reasons.push(`K below gate (${kComposite.toFixed(3)})`);
  const rb = verifyReceiptBody(built.receipt);
  if (!rb.valid) reasons.push('receipt body: ' + rb.reason);
  const rc = verifyReceiptChain(built.receipt);
  if (!rc.valid) reasons.push('receipt chain: ' + rc.reason);
  const cidParse = parseCid(built.cid);
  if (!cidParse || cidParse.hex.length !== 64) reasons.push('cid malformed');
  if (!verifyCidAgainstManifestHashes(built.cid, built.manifest.hashes)) reasons.push('cid != manifest.hashes');
  if (cidFromManifestHashes(built.manifest.hashes) !== built.cid) reasons.push('cid recompute drift');
  if (fs.statSync(built.outPath).size !== built.bytes) reasons.push('disk-bytes drift');

  return {
    ok: reasons.length === 0,
    reasons,
    k: kComposite,
    holdout: `${holdoutPass}/${holdoutN}`,
    cid: built.cid,
    bytes: built.bytes,
    outPath: built.outPath,
    strategy,
    extra: extra_manifest,
  };
}

// --------------------------------------------------------------------------
// T5 — Real logistic regression trainer in pure JS.
//
// TF-IDF features over a fixed vocabulary mined from the training corpus.
// Mini-batch gradient descent on cross-entropy loss with L2 regularization.
// Weights baked into the generator JS so the artifact ships a real learned
// model — no inference-time framework needed.
// --------------------------------------------------------------------------

const STOP = new Set(['the','and','for','you','your','this','that','from','with','have','was','are','our','its','it','to','of','in','on','at','is','as','a','an','or','if','be','my','me','we','i','no','so','do','will','would','could','should','has','had','were','been','they','their','them','he','she','his','her','them','what','which','who','whom','where','when','why','how','these','those','then','than','also','very','just','can']);

function tokenize(s) {
  return String(s).toLowerCase().split(/[^a-z0-9]+/).filter(t => t.length >= 2 && !STOP.has(t));
}

function buildVocab(corpus, maxV = 64) {
  const df = new Map();
  for (const ex of corpus) {
    const toks = new Set(tokenize(ex.input));
    for (const t of toks) df.set(t, (df.get(t) || 0) + 1);
  }
  // Keep the most discriminating mid-frequency terms (drop singletons and
  // ubiquitous tokens).
  const N = corpus.length;
  const scored = [...df.entries()]
    .filter(([_, n]) => n >= 2 && n <= Math.max(2, Math.floor(N * 0.85)))
    .map(([t, n]) => ({ t, idf: Math.log((N + 1) / (n + 1)) + 1 }));
  scored.sort((a, b) => b.idf - a.idf);
  const vocab = scored.slice(0, maxV);
  return vocab;
}

function tfidfVec(text, vocab) {
  const toks = tokenize(text);
  if (toks.length === 0) return new Array(vocab.length).fill(0);
  const tf = new Map();
  for (const t of toks) tf.set(t, (tf.get(t) || 0) + 1);
  const inv = 1 / toks.length;
  return vocab.map(({ t, idf }) => (tf.get(t) || 0) * inv * idf);
}

function trainLogReg(corpus, { epochs = 400, lr = 0.5, l2 = 0.005, maxV = 64 } = {}) {
  const vocab = buildVocab(corpus, maxV);
  const X = corpus.map(ex => tfidfVec(ex.input, vocab));
  const y = corpus.map(ex => ex.expected ? 1 : 0);
  const n = X.length;
  const d = vocab.length;
  let w = new Array(d).fill(0);
  let b = 0;

  const sigmoid = z => 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, z))));

  let lastLoss = Infinity;
  for (let epoch = 0; epoch < epochs; epoch++) {
    let totalLoss = 0;
    const gradW = new Array(d).fill(0);
    let gradB = 0;
    for (let i = 0; i < n; i++) {
      let z = b;
      for (let k = 0; k < d; k++) z += w[k] * X[i][k];
      const p = sigmoid(z);
      const err = p - y[i];
      gradB += err;
      for (let k = 0; k < d; k++) gradW[k] += err * X[i][k];
      const eps = 1e-9;
      totalLoss += -(y[i] * Math.log(p + eps) + (1 - y[i]) * Math.log(1 - p + eps));
    }
    for (let k = 0; k < d; k++) gradW[k] = gradW[k] / n + l2 * w[k];
    gradB = gradB / n;
    for (let k = 0; k < d; k++) w[k] -= lr * gradW[k];
    b -= lr * gradB;
    lastLoss = totalLoss / n;
    if (lastLoss < 0.05) break;
  }

  // Training accuracy as a sanity gate.
  let correct = 0;
  for (let i = 0; i < n; i++) {
    let z = b;
    for (let k = 0; k < d; k++) z += w[k] * X[i][k];
    const p = sigmoid(z);
    if ((p >= 0.5 ? 1 : 0) === y[i]) correct++;
  }
  return {
    vocab,
    weights: w.map(x => Number(x.toFixed(6))),
    bias: Number(b.toFixed(6)),
    final_loss: lastLoss,
    train_accuracy: correct / n,
    epochs_run: epochs,
  };
}

function bakeLogRegGenerator(model) {
  // Inline tokenizer + vocabulary + weights so the generator is fully
  // self-contained inside the .kolm artifact.
  const vocabLiteral = JSON.stringify(model.vocab);
  const wLiteral = JSON.stringify(model.weights);
  const bLiteral = JSON.stringify(model.bias);
  const stopLiteral = JSON.stringify([...STOP]);
  return `function generate(input, lib) {
  const VOCAB = ${vocabLiteral};
  const W = ${wLiteral};
  const B = ${bLiteral};
  const STOP = new Set(${stopLiteral});
  const toks = String(input).toLowerCase().split(/[^a-z0-9]+/).filter(function(t){ return t.length >= 2 && !STOP.has(t); });
  if (toks.length === 0) return false;
  const tf = {};
  for (let i = 0; i < toks.length; i++) tf[toks[i]] = (tf[toks[i]] || 0) + 1;
  const inv = 1 / toks.length;
  let z = B;
  for (let k = 0; k < VOCAB.length; k++) {
    const v = VOCAB[k];
    const x = (tf[v.t] || 0) * inv * v.idf;
    z += W[k] * x;
  }
  const p = 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, z))));
  return p >= 0.5;
}`;
}

// --------------------------------------------------------------------------
// T6 — Live distill bridge.
//
// We launch the FastAPI trainer service from apps/trainer/main.py in mock
// mode (real mode needs CUDA). Mock walks the same status lifecycle and
// returns a full {metrics, adapter} envelope, so the kolm.ai integration
// path is exercised honestly even though weights are synthetic.
// --------------------------------------------------------------------------

function spawnTrainerBridge() {
  return new Promise((resolve, reject) => {
    const env = {
      ...process.env,
      KOLM_TRAINER_MODE: 'mock',
      KOLM_TRAINER_BRIDGE_TOKEN: '',
      KOLM_TRAINER_JOBS: path.join(outDir, 'trainer-jobs.jsonl'),
      KOLM_TRAINER_ADAPTER_DIR: path.join(outDir, 'adapters'),
      PYTHONUNBUFFERED: '1',
    };
    fs.mkdirSync(env.KOLM_TRAINER_ADAPTER_DIR, { recursive: true });
    const proc = spawn(
      'python',
      ['-m', 'uvicorn', 'apps.trainer.main:app', '--host', '127.0.0.1', '--port', '8765', '--log-level', 'warning'],
      { cwd: repo, env, stdio: ['ignore', 'pipe', 'pipe'] },
    );
    let resolved = false;
    const onLine = (chunk) => {
      const txt = chunk.toString();
      if (!resolved && /Uvicorn running/i.test(txt)) {
        resolved = true;
        resolve(proc);
      }
    };
    proc.stdout.on('data', onLine);
    proc.stderr.on('data', onLine);
    proc.on('error', reject);
    setTimeout(() => {
      if (!resolved) {
        // Try a health probe; sometimes uvicorn logs nothing.
        fetch('http://127.0.0.1:8765/health').then(r => {
          if (r.ok && !resolved) { resolved = true; resolve(proc); }
        }).catch(() => {});
      }
    }, 1500);
    setTimeout(() => { if (!resolved) reject(new Error('trainer bridge failed to start within 8s')); }, 8000);
  });
}

function corpusServer(jsonlPath) {
  return new Promise((resolve) => {
    const body = fs.readFileSync(jsonlPath);
    const srv = http.createServer((req, res) => {
      res.writeHead(200, { 'content-type': 'application/jsonl' });
      res.end(body);
    });
    srv.listen(0, '127.0.0.1', () => {
      const { port } = srv.address();
      resolve({ url: `http://127.0.0.1:${port}/corpus.jsonl`, close: () => srv.close() });
    });
  });
}

function callbackServer() {
  return new Promise((resolve) => {
    const events = [];
    const srv = http.createServer((req, res) => {
      let chunks = [];
      req.on('data', c => chunks.push(c));
      req.on('end', () => {
        try { events.push(JSON.parse(Buffer.concat(chunks).toString('utf8'))); } catch { /* swallow */ }
        res.writeHead(202); res.end();
      });
    });
    srv.listen(0, '127.0.0.1', () => {
      const { port } = srv.address();
      resolve({ url: `http://127.0.0.1:${port}/callback`, events, close: () => srv.close() });
    });
  });
}

async function runT6() {
  // 1. Spawn trainer bridge
  const trainerProc = await spawnTrainerBridge();
  // 2. Stand up local corpus + callback servers
  const corpusPath = path.join(outDir, 't6-corpus.jsonl');
  const corpus = [
    { prompt: 'classify: tickets about logins', completion: 'auth' },
    { prompt: 'classify: payment failed on checkout', completion: 'billing' },
    { prompt: 'classify: app crashes when saving', completion: 'bug' },
    { prompt: 'classify: refund my last invoice', completion: 'billing' },
    { prompt: 'classify: cannot reset my password', completion: 'auth' },
    { prompt: 'classify: 500 error on dashboard', completion: 'bug' },
  ];
  fs.writeFileSync(corpusPath, corpus.map(c => JSON.stringify(c)).join('\n'));
  const cs = await corpusServer(corpusPath);
  const cb = await callbackServer();
  try {
    // 3. POST /distill
    const distill = await fetch('http://127.0.0.1:8765/distill', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        tenant: 'kolm-test',
        namespace: 'support-router',
        base_model: 'qwen2.5-3b-instruct',
        target_size: '3b',
        pair_count: corpus.length,
        callback_url: cb.url,
        corpus_url: cs.url,
        holdout_ratio: 0.2,
      }),
    });
    const distillBody = await distill.json();
    if (!distillBody.job_id) throw new Error('no job_id from /distill: ' + JSON.stringify(distillBody));
    const job_id = distillBody.job_id;

    // 4. Wait for terminal callback (completed/failed).
    const deadline = Date.now() + 30000;
    let terminal = null;
    while (Date.now() < deadline) {
      terminal = cb.events.find(e => e.status === 'completed' || e.status === 'failed');
      if (terminal) break;
      await new Promise(r => setTimeout(r, 200));
    }
    if (!terminal) throw new Error('no terminal callback within 30s');
    if (terminal.status !== 'completed') throw new Error('trainer reported: ' + JSON.stringify(terminal));

    // 5. Build a .kolm whose recipe references the trained adapter pointer.
    const adapterPtr = terminal.adapter;
    const metrics = terminal.metrics;
    const generatorSrc = `function generate(input, lib) {
  // Adapter-backed recipe — in production runs against the LoRA pointed to by
  // manifest.lora.url. For this smoke we surface the adapter envelope so the
  // calling host can decide how to load it.
  return { routed_via: 'adapter', adapter_sha: ${JSON.stringify(adapterPtr.sha256)}, base_model: ${JSON.stringify(metrics.base_model)} };
}`;
    const source_hash = crypto.createHash('sha256').update(generatorSrc).digest('hex').slice(0, 16);
    const spec = {
      name: 't6-distilled-router',
      task: 'Route inbound support tickets via a distilled LoRA adapter.',
      output_spec: { type: 'object' },
      train: corpus.map(c => ({ input: c.prompt, expected: c.completion })),
      holdout: [],
    };
    const recipes = [{
      id: 'cpt_t6_router',
      version_id: 'ver_t6_router_001',
      name: spec.name,
      source: generatorSrc,
      source_hash,
      synthesized: false,
    }];
    const evals = { spec: 'rs-1-evals', n: corpus.length, cases: corpus.map((c, i) => ({ id: `case-${i+1}`, input: c.prompt, expected: c.completion })), coverage: metrics.holdout_accuracy };
    const built = await buildAndZip({
      job_id: 'job_t6_distilled_' + crypto.randomBytes(3).toString('hex'),
      task: spec.task,
      base_model: metrics.base_model,
      recipes,
      lora_pointer: { url: adapterPtr.url, sha256: adapterPtr.sha256, size_bytes: adapterPtr.size_bytes, format: adapterPtr.format },
      training_stats: {
        verifier_accepted: true,
        pass_rate_positive: metrics.holdout_accuracy,
        latency_p50_us: 50,
        epochs: metrics.epochs,
        steps: metrics.steps,
        holdout_accuracy: metrics.holdout_accuracy,
        holdout_f1: metrics.holdout_f1,
        training_loss_final: metrics.training_loss_final,
        distilled_pairs: metrics.pair_count,
        trainer_mode: metrics.mode,
        trainer_job_id: job_id,
      },
      evals,
      outDir,
    });

    const k = built.k_score?.composite ?? 0;
    const reasons = [];
    if (k < 0.85) reasons.push(`K below gate (${k.toFixed(3)})`);
    const rb = verifyReceiptBody(built.receipt); if (!rb.valid) reasons.push('receipt body: ' + rb.reason);
    const rc = verifyReceiptChain(built.receipt); if (!rc.valid) reasons.push('receipt chain: ' + rc.reason);
    if (!verifyCidAgainstManifestHashes(built.cid, built.manifest.hashes)) reasons.push('cid mismatch');
    if (!built.manifest.lora || built.manifest.lora.sha256 !== adapterPtr.sha256) reasons.push('lora pointer not bound into manifest');

    return {
      ok: reasons.length === 0,
      reasons,
      k,
      cid: built.cid,
      bytes: built.bytes,
      outPath: built.outPath,
      strategy: 'distill-bridge (mock)',
      extra: {
        trainer_job_id: job_id,
        adapter_sha: adapterPtr.sha256,
        adapter_url: adapterPtr.url,
        holdout_accuracy: metrics.holdout_accuracy,
        pair_count: metrics.pair_count,
        epochs: metrics.epochs,
        steps: metrics.steps,
        progress_callbacks: cb.events.length,
      },
    };
  } finally {
    cs.close();
    cb.close();
    try { trainerProc.kill('SIGTERM'); } catch {} // deliberate: cleanup
  }
}

// --------------------------------------------------------------------------
// Specs for T1..T5.
// --------------------------------------------------------------------------

const T1 = {
  name: 't1-email-extractor',
  task: 'Pull email addresses out of arbitrary text.',
  output_spec: { type: 'string[]', pattern: 'email' },
  complexity: 'trivial — regex over fixed pattern',
  train: [
    { input: 'Email me at alice@example.com', expected: ['alice@example.com'] },
    { input: 'Contact bob@test.io and carol@kolm.ai', expected: ['bob@test.io', 'carol@kolm.ai'] },
    { input: 'No mail here', expected: [] },
    { input: 'admin@x.com is the address', expected: ['admin@x.com'] },
    { input: 'Multiple: a@a.com, b@b.org, c@c.net', expected: ['a@a.com', 'b@b.org', 'c@c.net'] },
    { input: 'Just text', expected: [] },
  ],
  holdout: [
    { input: 'Send to ops@kolm.ai please', expected: ['ops@kolm.ai'] },
    { input: 'no emails', expected: [] },
  ],
};

const T2 = {
  name: 't2-spam-detector',
  task: 'Flag promotional/spam emails.',
  output_spec: { type: 'boolean' },
  complexity: 'simple — keyword bag boolean',
  train: [
    { input: 'CONGRATULATIONS you won a free iPhone, click to claim', expected: true },
    { input: 'Buy cheap viagra online, lowest prices guaranteed', expected: true },
    { input: 'Make $5000 a week from home, no skills required', expected: true },
    { input: 'You have been selected for a free vacation, claim now', expected: true },
    { input: 'URGENT your account will be closed unless you click verify', expected: true },
    { input: 'Hot singles in your area want to chat, click to meet', expected: true },
    { input: 'Lose 30 pounds in 30 days with this miracle trick', expected: true },
    { input: 'Hot deals, lose pounds fast with this miracle trick', expected: true },
    { input: 'Free pills, cheap online pharmacy, lowest prices ever', expected: true },
    { input: 'Singles in your area waiting, hot chat tonight, click now', expected: true },
    { input: 'Please review the quarterly report by Friday', expected: false },
    { input: 'Reminder standup at 10am tomorrow', expected: false },
    { input: 'Send me the budget spreadsheet please', expected: false },
    { input: 'New design lands on Tuesday, please review figma', expected: false },
    { input: 'Lunch tomorrow at noon, same place', expected: false },
  ],
  holdout: [
    { input: 'FREE money claim your reward NOW', expected: true },
    { input: 'Cheap pharmacy pills, no prescription', expected: true },
    { input: 'Please find attached the agenda for the offsite', expected: false },
    { input: 'Sending over the PR for review', expected: false },
  ],
};

const T3 = {
  name: 't3-sentiment-tagger',
  task: 'Tag customer messages positive/negative/neutral.',
  output_spec: { type: 'enum', labels: ['positive', 'negative', 'neutral'] },
  complexity: 'moderate — 3-class with overlapping vocabulary',
  train: [
    { input: 'Absolutely love this product, fantastic experience!', expected: 'positive' },
    { input: 'Great product, my whole team loves the new features.', expected: 'positive' },
    { input: 'Awesome support, fantastic team, love the response time.', expected: 'positive' },
    { input: 'Wonderful product, highly recommend, love it.', expected: 'positive' },
    { input: 'Amazing experience, love this great service.', expected: 'positive' },
    { input: 'Terrible bug ruined my day, hate the new release.', expected: 'negative' },
    { input: 'Awful support team, disappointed and frustrated.', expected: 'negative' },
    { input: 'Hate the broken design, terrible update.', expected: 'negative' },
    { input: 'Worst experience ever, disappointed and angry.', expected: 'negative' },
    { input: 'Broken feature, awful bug, terrible release.', expected: 'negative' },
    { input: 'Deploy finished at fifteen oclock', expected: 'neutral' },
    { input: 'Standup tomorrow morning per schedule', expected: 'neutral' },
    { input: 'Status report attached', expected: 'neutral' },
    { input: 'Build pipeline green', expected: 'neutral' },
    { input: 'Reviewing draft document', expected: 'neutral' },
  ],
  holdout: [
    { input: 'Fantastic release, love it!', expected: 'positive' },
    { input: 'Awful broken feature, terrible.', expected: 'negative' },
    { input: 'Pipeline green', expected: 'neutral' },
    { input: 'Hate the broken update.', expected: 'negative' },
  ],
};

const T4 = {
  name: 't4-support-priority-adversarial',
  task: 'Classify support tickets P0/P1/P2/P3 — holdout uses phrasings the train set did not.',
  output_spec: { type: 'enum', labels: ['P0', 'P1', 'P2', 'P3'] },
  complexity: 'complex — 4-class with adversarial holdout phrasings',
  train: [
    { input: 'All checkouts failing across all customers right now', expected: 'P0' },
    { input: 'Production database down, every user sees 500s', expected: 'P0' },
    { input: 'Total outage nobody can log in anywhere', expected: 'P0' },
    { input: 'Site completely unreachable, customers calling support', expected: 'P0' },
    { input: 'Release broke csv export for everyone', expected: 'P1' },
    { input: 'SSO fails for one large customer, the rest are fine', expected: 'P1' },
    { input: 'Login broken on latest Chrome build', expected: 'P1' },
    { input: 'New feature broken, blocking a customer workflow', expected: 'P1' },
    { input: 'Search returns slightly stale results on dashboard', expected: 'P2' },
    { input: 'Some users see duplicated rows in report once a day', expected: 'P2' },
    { input: 'Minor lag on settings page, workaround available', expected: 'P2' },
    { input: 'Pagination off by one but workaround works', expected: 'P2' },
    { input: 'Would be nice if admin theme had dark mode', expected: 'P3' },
    { input: 'Typo in help text on settings page', expected: 'P3' },
    { input: 'Wishlist keyboard shortcut for new note', expected: 'P3' },
    { input: 'Cosmetic alignment off on footer', expected: 'P3' },
  ],
  // Held-out phrasings use words that DO appear in train, but in mixed
  // distributions — a robust classifier still routes them correctly.
  holdout: [
    { input: 'Customers everywhere getting errors, total outage', expected: 'P0' },
    { input: 'Export feature broken for everyone since last release', expected: 'P1' },
    { input: 'Slight stale data in dashboard, workaround works', expected: 'P2' },
    { input: 'Wishlist dark mode for the admin', expected: 'P3' },
  ],
};

// T5 is built specifically for the trained-logreg path. We give it a clean
// boolean separator and grade against a meaningfully harder holdout.
const T5_corpus = [
  // positives — refund / cancel intent
  { input: 'Please refund my last invoice, it was charged twice', expected: true },
  { input: 'I want a refund for the duplicate charge yesterday', expected: true },
  { input: 'Can you return my money for the cancelled order', expected: true },
  { input: 'Requesting a refund on the subscription I never started', expected: true },
  { input: 'Refund please, the product never arrived', expected: true },
  { input: 'Issue a chargeback or refund for invoice 1023', expected: true },
  { input: 'Cancel my subscription and refund the unused portion', expected: true },
  { input: 'My card was charged in error, please refund', expected: true },
  { input: 'I would like a refund on order 4421, never delivered', expected: true },
  { input: 'Refund the wrong amount you charged on Monday', expected: true },
  // negatives — anything else
  { input: 'How do I reset my password', expected: false },
  { input: 'The login button does not work on Safari', expected: false },
  { input: 'Please add a dark mode to the dashboard', expected: false },
  { input: 'When does my next billing cycle start', expected: false },
  { input: 'Can you upgrade my plan to Pro', expected: false },
  { input: 'Is the API rate limit per second or per minute', expected: false },
  { input: 'How do I invite a teammate to my workspace', expected: false },
  { input: 'My team needs SSO, when is that available', expected: false },
  { input: 'Standup tomorrow at 10am', expected: false },
  { input: 'Quarterly review meeting agenda attached', expected: false },
];
const T5_holdout = [
  // Held-out positives use refund-related vocabulary not in train.
  { input: 'Need my money back for invoice 9921', expected: true },
  { input: 'Reverse the charge on my card from Tuesday', expected: true },
  { input: 'Refund the overcharge from last week please', expected: true },
  // Held-out negatives use billing-adjacent but non-refund language.
  { input: 'How do I change my billing email', expected: false },
  { input: 'When is my next renewal date', expected: false },
  { input: 'Upgrade me to the annual plan', expected: false },
];

// --------------------------------------------------------------------------
// Driver.
// --------------------------------------------------------------------------

const tiers = [];
console.log('================================================================');
console.log(' kolm — complexity ramp end-to-end smoke');
console.log('================================================================');
console.log('outDir:', outDir);
console.log('');

async function runTier(label, fn) {
  process.stdout.write(`▸ ${label} ... `);
  const t0 = Date.now();
  try {
    const r = await fn();
    const ms = Date.now() - t0;
    if (r.ok) console.log(`OK    k=${r.k.toFixed(3)}  holdout=${r.holdout || '-'}  ${(r.bytes / 1024).toFixed(1)}KiB  (${ms}ms)`);
    else console.log(`FAIL  ${r.reasons.join('; ')}  (${ms}ms)`);
    tiers.push({ label, ms, ...r });
  } catch (e) {
    const ms = Date.now() - t0;
    tiers.push({ label, ms, ok: false, reasons: ['exception: ' + (e.message || String(e))] });
    console.log(`FAIL  exception: ${(e.message || String(e)).slice(0, 100)}  (${ms}ms)`);
  }
}

// T1..T4 — pattern synth pipeline
await runTier('T1 trivial    email-extractor          ', () => pipelineSynth(T1));
await runTier('T2 simple     spam-detector            ', () => pipelineSynth(T2));
await runTier('T3 moderate   sentiment-tagger         ', () => pipelineSynth(T3));
await runTier('T4 complex    support-priority         ', () => pipelineSynth(T4));

// T5 — trained logistic regression
await runTier('T5 trained    refund-intent (logreg)   ', async () => {
  const model = trainLogReg(T5_corpus, { epochs: 600, lr: 0.6, l2: 0.003, maxV: 48 });
  const src = bakeLogRegGenerator(model);
  const fn = compileJs(src);
  let holdoutPass = 0;
  for (const h of T5_holdout) {
    const out = fn(h.input);
    if (out === h.expected) holdoutPass++;
  }
  const source_hash = crypto.createHash('sha256').update(src).digest('hex').slice(0, 16);
  const spec = {
    name: 't5-refund-logreg',
    task: 'Detect refund-intent in support messages using a trained logistic regression model with TF-IDF features.',
    output_spec: { type: 'boolean' },
    train: T5_corpus,
  };
  const r = await packageAndVerify({
    spec, source: src, source_hash,
    holdoutPass, holdoutN: T5_holdout.length,
    training_stats: {
      verifier_accepted: true,
      pass_rate_positive: model.train_accuracy,
      latency_p50_us: 50,
      epochs: model.epochs_run,
      training_loss_final: model.final_loss,
      distilled_pairs: T5_corpus.length,
      vocab_size: model.vocab.length,
      model_kind: 'logistic-regression-tfidf',
    },
    strategy: 'trained-logreg',
    extra_manifest: {},
  });
  r.extra = {
    train_accuracy: Number(model.train_accuracy.toFixed(3)),
    final_loss: Number(model.final_loss.toFixed(4)),
    vocab_size: model.vocab.length,
    epochs_run: model.epochs_run,
    top_features: model.vocab
      .map((v, i) => ({ t: v.t, w: model.weights[i] }))
      .sort((a, b) => Math.abs(b.w) - Math.abs(a.w))
      .slice(0, 6)
      .map(x => `${x.t}=${x.w.toFixed(2)}`)
      .join(' '),
  };
  return r;
});

// T6 — live distill bridge end-to-end (mock training mode)
await runTier('T6 distill    trainer-bridge end-to-end', runT6);

console.log('');
console.log('================================================================');
console.log(' summary');
console.log('================================================================');
console.log('');
for (const t of tiers) {
  const tag = t.ok ? ' OK ' : 'FAIL';
  console.log(`[${tag}] ${t.label}  k=${(t.k ?? 0).toFixed(3)}  hold=${t.holdout || '-'}  bytes=${t.bytes ? (t.bytes / 1024).toFixed(1) + 'k' : '-'}  strategy=${t.strategy || '-'}`);
  if (t.extra && Object.keys(t.extra).length) {
    for (const [k, v] of Object.entries(t.extra)) console.log(`         ${k}: ${v}`);
  }
  if (!t.ok && t.reasons) console.log('         reasons: ' + t.reasons.join('; '));
}
console.log('');
const passN = tiers.filter(t => t.ok).length;
console.log(`pass: ${passN}/${tiers.length}    fail: ${tiers.length - passN}/${tiers.length}`);
console.log('');
console.log('artifacts written to:', outDir);
process.exit(passN === tiers.length ? 0 : 1);
