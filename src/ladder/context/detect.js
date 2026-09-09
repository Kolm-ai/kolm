// kolm code — command detection + tool-output parsers (spec 5.5). Owner C.
//
// Two jobs:
//   1. detectCommands(wt) — look at the manifests in a worktree and decide which
//      shell commands run the tests / the typechecker / the linter / coverage.
//   2. Parsers that turn those commands' output into STABLE IDS and test names,
//      so L1/L2 can compute a delta against the baseline instead of an absolute.
//
// Error ids deliberately exclude line/column for typecheck errors: a patch that
// shifts a file by three lines must not look like it introduced three errors.
// Lint ids keep the line because eslint rules are usually line-local and the
// rule id alone would collapse repeats.

import crypto from 'node:crypto';

const HASH_LEN = 8;

function hash(text) {
  return crypto.createHash('sha256').update(String(text)).digest('hex').slice(0, HASH_LEN);
}

function posix(p) {
  return String(p || '').replace(/\\/g, '/');
}

/** Make `p` relative to `root` (both may use either separator). */
export function relativize(p, root) {
  const file = posix(p);
  if (!root) return file.replace(/^\.\//, '');
  const base = posix(root).replace(/\/+$/, '');
  if (base && file.toLowerCase().startsWith(`${base.toLowerCase()}/`)) return file.slice(base.length + 1);
  return file.replace(/^\.\//, '');
}

async function readMaybe(wt, rel) {
  try {
    const text = await wt.readFile(rel);
    return typeof text === 'string' ? text : null;
  } catch {
    return null;
  }
}

async function readJsonMaybe(wt, rel) {
  const text = await readMaybe(wt, rel);
  if (text === null) return null;
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

async function exists(wt, rel) {
  return (await readMaybe(wt, rel)) !== null;
}

const ESLINT_CONFIGS = [
  'eslint.config.js', 'eslint.config.mjs', 'eslint.config.cjs', 'eslint.config.ts',
  '.eslintrc', '.eslintrc.js', '.eslintrc.cjs', '.eslintrc.json', '.eslintrc.yml', '.eslintrc.yaml',
];

/**
 * Which test runner a command string invokes — used as a parser hint.
 * @param {string} cmd
 * @returns {'node'|'jest'|'vitest'|'pytest'|'go'|'cargo'|'unknown'}
 */
export function detectRunner(cmd) {
  const c = String(cmd || '').toLowerCase();
  if (!c) return 'unknown';
  if (/\bvitest\b/.test(c)) return 'vitest';
  if (/\bjest\b/.test(c)) return 'jest';
  if (/\bpytest\b/.test(c) || /python\s+-m\s+pytest/.test(c)) return 'pytest';
  if (/\bgo\s+test\b/.test(c)) return 'go';
  if (/\bcargo\s+(test|nextest)\b/.test(c)) return 'cargo';
  if (/node\s+--test/.test(c) || /\bnode:test\b/.test(c)) return 'node';
  if (/\bnpm\s+(run\s+)?test\b/.test(c) || /\b(pnpm|yarn|bun)\s+(run\s+)?test\b/.test(c)) return 'node';
  return 'unknown';
}

/**
 * Detect the commands this repo uses.
 * @param {object} wt worktree (needs readFile; listFiles is optional)
 * @returns {Promise<{test:string|null, typecheck:string|null, lint:string|null,
 *                    coverage:string|null, language:string|null, packageManager:string|null}>}
 */
export async function detectCommands(wt) {
  const out = { test: null, typecheck: null, lint: null, coverage: null, language: null, packageManager: null };

  const pkg = await readJsonMaybe(wt, 'package.json');
  const pyproject = await readMaybe(wt, 'pyproject.toml');
  const pytestIni = await readMaybe(wt, 'pytest.ini');
  const setupCfg = await readMaybe(wt, 'setup.cfg');
  const goMod = await readMaybe(wt, 'go.mod');
  const cargoToml = await readMaybe(wt, 'Cargo.toml');

  if (pkg) {
    out.language = (await exists(wt, 'tsconfig.json')) ? 'typescript' : 'javascript';
    const hasPnpm = await exists(wt, 'pnpm-lock.yaml');
    const hasYarn = await exists(wt, 'yarn.lock');
    const hasBun = await exists(wt, 'bun.lockb');
    out.packageManager = hasPnpm ? 'pnpm' : hasYarn ? 'yarn' : hasBun ? 'bun' : 'npm';
    const run = out.packageManager === 'npm' ? 'npm run' : `${out.packageManager} run`;
    const scripts = (pkg.scripts && typeof pkg.scripts === 'object') ? pkg.scripts : {};
    const deps = { ...(pkg.dependencies || {}), ...(pkg.devDependencies || {}) };

    if (typeof scripts.test === 'string' && scripts.test.trim()) {
      out.test = out.packageManager === 'npm' ? 'npm test --silent' : `${out.packageManager} test`;
    } else if (deps.vitest) {
      out.test = 'npx vitest run';
    } else if (deps.jest) {
      out.test = 'npx jest';
    } else {
      out.test = 'node --test';
    }

    if (await exists(wt, 'tsconfig.json')) out.typecheck = 'npx tsc --noEmit -p .';
    else if (typeof scripts.typecheck === 'string') out.typecheck = `${run} typecheck`;

    for (const cfg of ESLINT_CONFIGS) {
      if (await exists(wt, cfg)) {
        out.lint = 'npx eslint . -f json';
        break;
      }
    }
    if (!out.lint && (pkg.eslintConfig || deps.eslint)) out.lint = 'npx eslint . -f json';

    if (typeof scripts.coverage === 'string' && scripts.coverage.trim()) {
      out.coverage = out.packageManager === 'npm' ? 'npm run coverage --silent' : `${run} coverage`;
    } else if (deps.c8) {
      out.coverage = 'npx c8 --reporter=json-summary --reporter=text npm test --silent';
    } else if (deps.nyc) {
      out.coverage = 'npx nyc --reporter=json-summary npm test --silent';
    } else if (detectRunner(out.test) === 'node') {
      out.coverage = 'node --test --experimental-test-coverage';
    }
    return out;
  }

  if (pyproject || pytestIni || setupCfg) {
    out.language = 'python';
    out.packageManager = pyproject && /\[tool\.poetry\]/.test(pyproject) ? 'poetry'
      : pyproject && /\[tool\.uv\]/.test(pyproject) ? 'uv' : 'pip';
    out.test = 'pytest -q';
    const wantsPyright = (pyproject && /\[tool\.pyright\]/.test(pyproject)) || (await exists(wt, 'pyrightconfig.json'));
    const wantsMypy = (pyproject && /\[tool\.mypy\]/.test(pyproject)) || (setupCfg && /\[mypy\]/.test(setupCfg)) || (await exists(wt, 'mypy.ini'));
    if (wantsPyright) out.typecheck = 'pyright --outputjson';
    else if (wantsMypy) out.typecheck = 'mypy .';
    const wantsRuff = (pyproject && /\[tool\.ruff\]/.test(pyproject)) || (await exists(wt, 'ruff.toml')) || (await exists(wt, '.ruff.toml'));
    if (wantsRuff) out.lint = 'ruff check --output-format json';
    if (pyproject && /pytest-cov|coverage/.test(pyproject)) out.coverage = 'pytest -q --cov --cov-report=json';
    return out;
  }

  if (goMod) {
    out.language = 'go';
    out.packageManager = 'go';
    out.test = 'go test ./...';
    out.typecheck = 'go vet ./...';
    out.coverage = 'go test ./... -cover';
    return out;
  }

  if (cargoToml) {
    out.language = 'rust';
    out.packageManager = 'cargo';
    out.test = 'cargo test';
    out.typecheck = 'cargo check';
    return out;
  }

  return out;
}

// --------------------------------------------------------------------- tests

const TICK_PASS = /^[✔✓√]\s+(.+)$/;
const TICK_FAIL = /^[✖✕×✗]\s+(.+)$/;
const TAP_OK = /^ok\s+(\d+)\s*-?\s*(.*)$/;
const TAP_NOT_OK = /^not ok\s+(\d+)\s*-?\s*(.*)$/;
const NODE_SUMMARY = /^[#ℹ]\s*(tests|pass|fail|skipped|todo|cancelled)\s+(\d+)\s*$/;
const GO_FAIL = /^---\s*FAIL:\s*(\S+)/;
const GO_PASS = /^---\s*PASS:\s*(\S+)/;
const GO_PKG_OK = /^ok\s+\S+\s+[\d.]+s/;
const PYTEST_FAILED = /^(?:FAILED|ERROR)\s+(\S+)/;
const CARGO_LINE = /^test\s+(\S+)\s+\.\.\.\s+(ok|FAILED|ignored)\b/;

function stripTiming(name) {
  return String(name)
    .replace(/\s*\(\d+(?:\.\d+)?\s*m?s\)\s*$/, '')
    .replace(/\s*#\s*(SKIP|TODO).*$/i, '')
    .trim();
}

// node --test closes its run with a summary block and then a header line
// ("✖ failing tests:") before listing the tests that failed. The header
// carries the same tick glyph as a failing test, so without this guard the
// header itself lands in the failing-test list and shows up in an l2 reason as
// a test named "failing tests:".
const RUNNER_NOISE = [
  /^(failing|passing|cancelled|skipped|todo)\s+tests?\s*:?$/i,
  /^(tests|suites|pass|fail|cancelled|skipped|todo|duration_ms)\s+[\d.]+$/i,
  /^test\s+at\s+\S+:\d+:\d+$/i,
  /^[-=*_\s:]*$/,
];

/**
 * True when a parsed "test name" is really a summary or header line.
 * @param {string} name
 * @returns {boolean}
 */
export function isRunnerNoise(name) {
  const clean = stripTiming(name);
  if (!clean) return true;
  return RUNNER_NOISE.some((re) => re.test(clean));
}

/**
 * Parse a test runner's combined stdout+stderr.
 * Handles: node --test (spec + TAP), jest, vitest, pytest, go test, cargo test.
 * @param {string} text
 * @param {string} [runnerHint] from detectRunner()
 * @returns {{passed:number, failed:number, failing:string[], parsed:boolean}}
 */
export function parseTestOutput(text, runnerHint) {
  const lines = String(text ?? '').split(/\r?\n/);
  const failing = [];
  let countPass = 0;
  let countFail = 0;
  let goPackages = 0;
  let parsed = false;
  let sumPass = null;
  let sumFail = null;
  let sumSource = null; // 'node' | 'jsts' | 'pytest' | 'cargo'

  const addFailing = (name) => {
    if (isRunnerNoise(name)) return;
    const clean = stripTiming(name);
    if (clean && !failing.includes(clean)) failing.push(clean);
  };
  const setSummary = (source, pass, fail) => {
    // node's own counters win over a framework banner; first writer of a source wins.
    const rank = { node: 3, cargo: 2, jsts: 2, pytest: 1 };
    if (sumSource && rank[sumSource] > rank[source]) return;
    sumSource = source;
    if (pass !== null && pass !== undefined) sumPass = pass;
    if (fail !== null && fail !== undefined) sumFail = fail;
  };

  for (const raw of lines) {
    const line = raw.trim();
    if (!line) continue;

    const nodeSummary = NODE_SUMMARY.exec(line);
    if (nodeSummary) {
      parsed = true;
      if (nodeSummary[1] === 'pass') setSummary('node', Number(nodeSummary[2]), sumFail);
      else if (nodeSummary[1] === 'fail') setSummary('node', sumPass, Number(nodeSummary[2]));
      continue;
    }

    let m = TICK_FAIL.exec(line);
    if (m) {
      parsed = true;
      if (!isRunnerNoise(m[1])) {
        countFail += 1;
        addFailing(m[1]);
      }
      continue;
    }
    m = TICK_PASS.exec(line);
    if (m) {
      parsed = true;
      if (!isRunnerNoise(m[1])) countPass += 1;
      continue;
    }
    m = TAP_NOT_OK.exec(line);
    if (m) {
      parsed = true;
      countFail += 1;
      addFailing(m[2] || `test ${m[1]}`);
      continue;
    }
    m = TAP_OK.exec(line);
    if (m) {
      parsed = true;
      countPass += 1;
      continue;
    }
    m = GO_FAIL.exec(line);
    if (m) {
      parsed = true;
      countFail += 1;
      addFailing(m[1]);
      continue;
    }
    m = GO_PASS.exec(line);
    if (m) {
      parsed = true;
      countPass += 1;
      continue;
    }
    if (GO_PKG_OK.test(line)) {
      parsed = true;
      goPackages += 1;
      continue;
    }
    m = PYTEST_FAILED.exec(line);
    if (m) {
      parsed = true;
      countFail += 1;
      addFailing(m[1]);
      continue;
    }
    m = CARGO_LINE.exec(line);
    if (m) {
      parsed = true;
      if (m[2] === 'ok') countPass += 1;
      else if (m[2] === 'FAILED') {
        countFail += 1;
        addFailing(m[1]);
      }
      continue;
    }
    if (/^Tests[:\s]/.test(line)) {
      // jest "Tests:  1 failed, 3 passed, 4 total" / vitest "Tests  1 failed | 3 passed (4)"
      const p = /(\d+)\s+passed/.exec(line);
      const f = /(\d+)\s+failed/.exec(line);
      if (p || f) {
        parsed = true;
        setSummary('jsts', p ? Number(p[1]) : 0, f ? Number(f[1]) : 0);
      }
      continue;
    }
    if (/^test result:/.test(line)) {
      const p = /(\d+)\s+passed/.exec(line);
      const f = /(\d+)\s+failed/.exec(line);
      if (p || f) {
        parsed = true;
        setSummary('cargo', p ? Number(p[1]) : 0, f ? Number(f[1]) : 0);
      }
      continue;
    }
    if (/\d+\s+(passed|failed|error)/.test(line) && /\bin\b\s+[\d.]+s|={3,}/.test(line)) {
      // pytest "===== 1 failed, 3 passed in 0.12s ====="
      const p = /(\d+)\s+passed/.exec(line);
      const f = /(\d+)\s+(?:failed|error)/.exec(line);
      if (p || f) {
        parsed = true;
        setSummary('pytest', p ? Number(p[1]) : 0, f ? Number(f[1]) : 0);
      }
      continue;
    }
  }

  if (countPass === 0 && goPackages > 0) countPass = goPackages;
  const passed = sumPass === null || sumPass === undefined ? countPass : sumPass;
  const failed = sumFail === null || sumFail === undefined ? countFail : sumFail;
  void runnerHint; // hint is advisory; every format above is matched unambiguously
  return { passed, failed, failing, parsed };
}

// ----------------------------------------------------------------- typecheck

const TSC_RE = /^(.+?)\((\d+),(\d+)\):\s*error\s+(TS\d+):\s*(.*)$/;
const MYPY_RE = /^(.+?):(\d+):(?:\d+:)?\s*error:\s*(.*?)(?:\s+\[([\w-]+)\])?$/;
const PYRIGHT_TEXT_RE = /^(.+?):(\d+):(\d+)\s*-\s*error:\s*(.*)$/;
const GOVET_RE = /^(.+?\.go):(\d+):(?:(\d+):)?\s*(.+)$/;
const CARGO_ERR_RE = /^error(?:\[(E\d+)\])?:\s*(.*)$/;
const CARGO_LOC_RE = /^\s*-->\s*(.+?):(\d+):(\d+)/;

/**
 * Stable ids for typechecker errors.
 * tsc     -> `path:TSxxxx:<hash of message>`
 * mypy    -> `path:<code|mypy>:<hash>`
 * pyright -> `path:<rule|pyright>:<hash>`
 * go vet  -> `path:vet:<hash>`
 * cargo   -> `path:<Ennnn|cargo>:<hash>`
 * @param {string} text
 * @param {string} [tool] 'tsc' | 'mypy' | 'pyright' | 'go' | 'cargo' (inferred when omitted)
 * @param {{root?:string}} [opts]
 * @returns {string[]}
 */
export function parseTypecheckErrors(text, tool, opts = {}) {
  const root = opts.root;
  const src = String(text ?? '');
  const ids = [];
  const add = (file, code, message) => {
    const id = `${relativize(file, root)}:${code}:${hash(message)}`;
    if (!ids.includes(id)) ids.push(id);
  };

  // pyright --outputjson
  if (!tool || tool === 'pyright') {
    const start = src.indexOf('{');
    if (start >= 0) {
      try {
        const doc = JSON.parse(src.slice(start));
        if (Array.isArray(doc.generalDiagnostics)) {
          for (const d of doc.generalDiagnostics) {
            if (d.severity && d.severity !== 'error') continue;
            add(d.file || '', d.rule || 'pyright', d.message || '');
          }
          return ids;
        }
      } catch {
        // fall through to the text parsers
      }
    }
  }

  const lines = src.split(/\r?\n/);
  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];
    const trimmed = line.trim();
    if (!trimmed) continue;

    let m = TSC_RE.exec(trimmed);
    if (m) {
      add(m[1], m[4], m[5]);
      continue;
    }
    m = PYRIGHT_TEXT_RE.exec(trimmed);
    if (m) {
      add(m[1], 'pyright', m[4]);
      continue;
    }
    m = MYPY_RE.exec(trimmed);
    if (m) {
      add(m[1], m[4] || 'mypy', m[3]);
      continue;
    }
    m = CARGO_ERR_RE.exec(trimmed);
    if (m) {
      const loc = CARGO_LOC_RE.exec(lines[i + 1] || '');
      add(loc ? loc[1] : 'crate', m[1] || 'cargo', m[2]);
      continue;
    }
    m = GOVET_RE.exec(trimmed);
    if (m) {
      add(m[1], 'vet', m[4]);
      continue;
    }
  }
  return ids;
}

// ---------------------------------------------------------------------- lint

/**
 * Stable ids for lint errors: `path:ruleId:line`.
 * eslint `-f json` and ruff `--output-format json` are both understood; anything
 * else falls back to a generic `path:line:col: message` text scrape.
 * Only error-severity findings count (warnings do not gate a patch).
 * @param {string} text
 * @param {string} [tool] 'eslint' | 'ruff' (inferred when omitted)
 * @param {{root?:string}} [opts]
 * @returns {string[]}
 */
export function parseLintErrors(text, tool, opts = {}) {
  const root = opts.root;
  const src = String(text ?? '');
  const ids = [];
  const add = (id) => {
    if (id && !ids.includes(id)) ids.push(id);
  };

  const start = src.search(/[[{]/);
  if (start >= 0) {
    try {
      const doc = JSON.parse(src.slice(start));
      if (Array.isArray(doc)) {
        if (doc.length === 0) return ids;
        if (doc[0] && Array.isArray(doc[0].messages)) {
          // eslint
          for (const file of doc) {
            for (const msg of file.messages || []) {
              if (Number(msg.severity) < 2) continue;
              add(`${relativize(file.filePath || '', root)}:${msg.ruleId || 'fatal'}:${msg.line ?? 0}`);
            }
          }
          return ids;
        }
        if (doc[0] && (doc[0].filename || doc[0].code)) {
          // ruff
          for (const d of doc) {
            const line = d.location?.row ?? d.location?.line ?? 0;
            add(`${relativize(d.filename || '', root)}:${d.code || 'ruff'}:${line}`);
          }
          return ids;
        }
      }
    } catch {
      // not JSON — fall through
    }
  }

  for (const raw of src.split(/\r?\n/)) {
    const m = /^(.+?):(\d+):(\d+):\s*(?:error\s+)?(.+)$/.exec(raw.trim());
    if (!m) continue;
    add(`${relativize(m[1], root)}:${hash(m[4])}:${m[2]}`);
  }
  return ids;
}

export default {
  detectCommands,
  detectRunner,
  parseTestOutput,
  parseTypecheckErrors,
  parseLintErrors,
  relativize,
};
