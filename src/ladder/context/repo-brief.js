// kolm code — decode-once repo brief, cached per SHA (spec 5.5). Owner C.
//
// Every candidate's coder and every reviewer needs the same orientation text:
// what files exist, what language this is, what the manifests declare, what the
// README says, how you run the tests, and what each source file exports. That is
// expensive to gather (one listFiles + N readFiles) and identical for every
// candidate of a run, so it is computed once per (repoKey, baseSha) and cached
// in the store's context_cache table.

import { detectCommands } from './detect.js';

const VENDOR_DIRS = new Set([
  'node_modules', '.git', 'dist', 'build', 'out', 'coverage', 'vendor', 'target',
  '__pycache__', '.venv', 'venv', '.next', '.nuxt', '.cache', '.tox', '.mypy_cache',
  '.pytest_cache', '.gradle', 'Pods', 'bower_components', '.terraform',
]);

const SOURCE_EXT = new Set(['.js', '.mjs', '.cjs', '.jsx', '.ts', '.tsx', '.py', '.go', '.rs', '.rb', '.java', '.kt', '.php', '.cs', '.swift', '.c', '.h', '.cc', '.cpp', '.hpp']);

const TREE_CAP = 400;
const EXPORT_FILE_CAP = 60;
const EXPORT_NAME_CAP = 20;
const README_BYTES = 2048;
export const BRIEF_BYTE_CAP = 12 * 1024;

const EXPORT_PATTERNS = [
  /^\s*export\s+(?:async\s+)?(?:function|class|const|let|var)\s+([A-Za-z_$][\w$]*)/,
  /^\s*export\s+default\s+(?:async\s+)?(?:function|class)\s+([A-Za-z_$][\w$]*)/,
  /^\s*(?:async\s+)?def\s+([A-Za-z_][\w]*)\s*\(/,
  /^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_][\w]*)\s*\(/,
  /^\s*pub\s+(?:async\s+)?fn\s+([A-Za-z_][\w]*)/,
];
const EXPORT_LIST_RE = /^\s*export\s*\{([^}]*)\}/;

function posix(p) {
  return String(p || '').replace(/\\/g, '/');
}

function isVendored(file) {
  return posix(file).split('/').some((seg) => VENDOR_DIRS.has(seg));
}

function extOf(file) {
  const base = posix(file).split('/').pop() || '';
  const dot = base.lastIndexOf('.');
  return dot > 0 ? base.slice(dot).toLowerCase() : '';
}

async function readMaybe(wt, rel) {
  try {
    const text = await wt.readFile(rel);
    return typeof text === 'string' ? text : null;
  } catch {
    return null;
  }
}

function repoKeyFor(wt, opts = {}) {
  return String(opts.repoKey || wt?.repoKey || wt?.repoPath || wt?.path || 'repo');
}

function parseExports(text) {
  const names = [];
  for (const line of String(text).split(/\r?\n/)) {
    if (names.length >= EXPORT_NAME_CAP) break;
    const list = EXPORT_LIST_RE.exec(line);
    if (list) {
      for (const part of list[1].split(',')) {
        const name = part.trim().split(/\s+as\s+/).pop().trim();
        if (name && !names.includes(name) && names.length < EXPORT_NAME_CAP) names.push(name);
      }
      continue;
    }
    for (const re of EXPORT_PATTERNS) {
      const m = re.exec(line);
      if (m && !names.includes(m[1])) {
        names.push(m[1]);
        break;
      }
    }
  }
  return names;
}

async function manifestSummary(wt) {
  const manifests = {};
  const pkgText = await readMaybe(wt, 'package.json');
  if (pkgText) {
    try {
      const pkg = JSON.parse(pkgText);
      manifests['package.json'] = {
        name: pkg.name ?? null,
        version: pkg.version ?? null,
        type: pkg.type ?? null,
        scripts: pkg.scripts && typeof pkg.scripts === 'object' ? pkg.scripts : {},
        dependencies: Object.keys(pkg.dependencies || {}),
        devDependencies: Object.keys(pkg.devDependencies || {}),
      };
    } catch {
      manifests['package.json'] = { parse_error: true };
    }
  }
  const pyproject = await readMaybe(wt, 'pyproject.toml');
  if (pyproject) {
    const name = /^\s*name\s*=\s*["']([^"']+)["']/m.exec(pyproject);
    manifests['pyproject.toml'] = {
      name: name ? name[1] : null,
      tools: [...pyproject.matchAll(/^\s*\[tool\.([\w-]+)\]/gm)].map((m) => m[1]),
    };
  }
  const goMod = await readMaybe(wt, 'go.mod');
  if (goMod) {
    const mod = /^module\s+(\S+)/m.exec(goMod);
    const go = /^go\s+(\S+)/m.exec(goMod);
    manifests['go.mod'] = { module: mod ? mod[1] : null, go: go ? go[1] : null };
  }
  const cargo = await readMaybe(wt, 'Cargo.toml');
  if (cargo) {
    const name = /^\s*name\s*=\s*["']([^"']+)["']/m.exec(cargo);
    manifests['Cargo.toml'] = { name: name ? name[1] : null };
  }
  return manifests;
}

/**
 * Build (or fetch from the cache) the repo brief for one SHA.
 * @param {object|null} store openStore() result, or null to skip caching
 * @param {object} wt worktree
 * @param {string} sha base sha the brief describes
 * @param {{repoKey?:string, commands?:object, force?:boolean}} [opts]
 * @returns {Promise<object>} brief
 */
export async function repoBrief(store, wt, sha, opts = {}) {
  const repoKey = repoKeyFor(wt, opts);
  if (store && !opts.force) {
    const cached = store.cacheGet(repoKey, sha, 'brief');
    if (cached) return cached;
  }

  let files = [];
  try {
    files = (await wt.listFiles({ limit: 5000 })) || [];
  } catch {
    files = [];
  }
  const kept = files.map(posix).filter((f) => f && !isVendored(f)).sort();

  const languages = {};
  for (const f of kept) {
    const ext = extOf(f) || '(none)';
    languages[ext] = (languages[ext] || 0) + 1;
  }

  const tree = kept.slice(0, TREE_CAP);

  let readmeHead = null;
  for (const name of ['README.md', 'readme.md', 'README.rst', 'README.txt', 'README']) {
    const text = await readMaybe(wt, name);
    if (text !== null) {
      readmeHead = text.slice(0, README_BYTES);
      break;
    }
  }

  const manifests = await manifestSummary(wt);
  const commands = opts.commands || (await detectCommands(wt));

  const exports = {};
  const sourceFiles = kept.filter((f) => SOURCE_EXT.has(extOf(f))).slice(0, EXPORT_FILE_CAP);
  for (const f of sourceFiles) {
    const text = await readMaybe(wt, f);
    if (text === null) continue;
    const names = parseExports(text);
    if (names.length > 0) exports[f] = names;
  }

  const brief = {
    sha: sha ?? null,
    generatedAt: new Date().toISOString(),
    tree,
    languages,
    manifests,
    readmeHead,
    commands,
    exports,
    bytes: 0,
  };
  brief.bytes = Buffer.byteLength(JSON.stringify(brief), 'utf8');

  // Trim to the 12 KB target: exports first (most redundant), then the tree.
  const exportKeys = Object.keys(brief.exports);
  while (brief.bytes > BRIEF_BYTE_CAP && exportKeys.length > 0) {
    delete brief.exports[exportKeys.pop()];
    brief.bytes = Buffer.byteLength(JSON.stringify(brief), 'utf8');
  }
  while (brief.bytes > BRIEF_BYTE_CAP && brief.tree.length > 20) {
    brief.tree = brief.tree.slice(0, Math.max(20, Math.floor(brief.tree.length * 0.75)));
    brief.bytes = Buffer.byteLength(JSON.stringify(brief), 'utf8');
  }
  if (brief.bytes > BRIEF_BYTE_CAP && brief.readmeHead) {
    brief.readmeHead = brief.readmeHead.slice(0, 512);
    brief.bytes = Buffer.byteLength(JSON.stringify(brief), 'utf8');
  }

  if (store) store.cachePut(repoKey, sha, 'brief', brief);
  return brief;
}

/**
 * Render the brief for a model prompt. Always <= BRIEF_BYTE_CAP bytes.
 * @param {object} brief
 * @returns {string}
 */
export function briefToPrompt(brief) {
  if (!brief) return '';
  const out = [];
  out.push(`REPO BRIEF (base ${brief.sha || 'unknown'})`);
  const langs = Object.entries(brief.languages || {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .map(([ext, n]) => `${ext} ${n}`)
    .join(', ');
  if (langs) out.push(`languages: ${langs}`);
  const cmd = brief.commands || {};
  out.push(`commands: test=${cmd.test || '-'} typecheck=${cmd.typecheck || '-'} lint=${cmd.lint || '-'} coverage=${cmd.coverage || '-'}`);
  for (const [name, m] of Object.entries(brief.manifests || {})) {
    if (name === 'package.json') {
      const scripts = Object.keys(m.scripts || {}).slice(0, 12).join(', ');
      out.push(`package.json: ${m.name || '?'}@${m.version || '?'} scripts[${scripts}] deps ${(m.dependencies || []).length}+${(m.devDependencies || []).length}`);
    } else {
      out.push(`${name}: ${JSON.stringify(m)}`);
    }
  }
  if (brief.readmeHead) {
    out.push('README (head):');
    out.push(brief.readmeHead.slice(0, 1200));
  }
  out.push(`files (${(brief.tree || []).length} shown):`);
  out.push((brief.tree || []).join('\n'));
  const exportEntries = Object.entries(brief.exports || {});
  if (exportEntries.length > 0) {
    out.push('exports:');
    for (const [file, names] of exportEntries) out.push(`  ${file}: ${names.join(', ')}`);
  }
  let text = out.join('\n');
  if (Buffer.byteLength(text, 'utf8') > BRIEF_BYTE_CAP) {
    const marker = '\n[brief truncated]';
    // -8 of slack: cutting mid-codepoint can re-encode as a 3-byte U+FFFD.
    const room = BRIEF_BYTE_CAP - Buffer.byteLength(marker, 'utf8') - 8;
    text = Buffer.from(text, 'utf8').subarray(0, room).toString('utf8') + marker;
  }
  return text;
}

export default { repoBrief, briefToPrompt, BRIEF_BYTE_CAP };
