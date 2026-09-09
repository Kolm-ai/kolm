// kolm code - decode-once repo brief, cached per SHA (spec 5.5). Owner C.
//
// Every candidate's coder and every reviewer needs the same orientation text:
// what files exist, what language this is, what the manifests declare, what the
// README says, how you run the tests, and what each source file exports. That is
// expensive to gather (one listFiles + N readFiles) and identical for every
// candidate of a run, so it is computed once per (repoKey, baseSha) and cached
// in the store's context_cache table.
//
// SOTA spec 1.3 adds three optional attachments the orchestrator hangs on the
// brief object - `instructions` (1.2), `lessons` (1.5) and `retrieval` (1.1) -
// so every consumer that already renders briefToPrompt(brief) picks them up
// with no change of its own. A brief without them renders byte for byte as it
// always did; tests pin that.

import { detectCommands } from './detect.js';
import { isVendored, retrievalToPrompt } from './retrieve.js';
import { instructionsToPrompt } from './instructions.js';
import { lessonsToPrompt } from '../lessons.js';

const SOURCE_EXT = new Set(['.js', '.mjs', '.cjs', '.jsx', '.ts', '.tsx', '.py', '.go', '.rs', '.rb', '.java', '.kt', '.php', '.cs', '.swift', '.c', '.h', '.cc', '.cpp', '.hpp']);

const TREE_CAP = 400;
const EXPORT_FILE_CAP = 60;
const EXPORT_NAME_CAP = 20;
const README_BYTES = 2048;
const TEST_DIR_CAP = 20;
export const BRIEF_BYTE_CAP = 12 * 1024;

// ---------------------------------------------------------------- languages

/**
 * Extension -> language name. The brief reports languages, not file suffixes,
 * so `.mjs` and `.cjs` do not read as two more languages nobody uses.
 */
const LANGUAGE_BY_EXT = Object.freeze({
  '.js': 'js', '.mjs': 'js', '.cjs': 'js', '.jsx': 'js',
  '.ts': 'ts', '.tsx': 'ts', '.mts': 'ts', '.cts': 'ts',
  '.py': 'py', '.pyi': 'py',
  '.go': 'go',
  '.rs': 'rs',
  '.rb': 'rb',
  '.java': 'java', '.kt': 'kt', '.kts': 'kt', '.cs': 'cs',
  '.php': 'php', '.swift': 'swift',
  '.c': 'c', '.h': 'c',
  '.cc': 'cpp', '.cpp': 'cpp', '.cxx': 'cpp', '.hpp': 'cpp', '.hh': 'cpp',
  '.sh': 'sh', '.bash': 'sh', '.ps1': 'ps1',
  '.md': 'md', '.rst': 'rst', '.txt': 'txt',
  '.json': 'json', '.yml': 'yaml', '.yaml': 'yaml', '.toml': 'toml',
  '.html': 'html', '.css': 'css', '.scss': 'css', '.sql': 'sql',
});

/** Directory names that mean "the tests live under here". */
const TEST_DIR_SEGMENTS = new Set(['test', 'tests', '__tests__', 'spec', 'specs', 'testing']);
const TEST_FILE_RE = /(?:^|\.)(?:test|spec)\.[A-Za-z0-9]+$|^test_[^/]*\.[A-Za-z0-9]+$|_test\.[A-Za-z0-9]+$/;

// ------------------------------------------------------------ declarations

const JS_PATTERNS = [
  /^\s*export\s+(?:async\s+)?(?:function|class|const|let|var)\s+([A-Za-z_$][\w$]*)/,
  /^\s*export\s+default\s+(?:async\s+)?(?:function|class)\s+([A-Za-z_$][\w$]*)/,
];
const PY_PATTERNS = [
  /^\s*(?:async\s+)?def\s+([A-Za-z_][\w]*)\s*\(/,
  /^\s*class\s+([A-Za-z_][\w]*)/,
];
const GO_PATTERNS = [
  /^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_][\w]*)\s*\(/,
  /^\s*type\s+([A-Za-z_][\w]*)\s+/,
];
const RS_PATTERNS = [
  /^\s*pub\s+(?:async\s+)?fn\s+([A-Za-z_][\w]*)/,
  /^\s*pub\s+struct\s+([A-Za-z_][\w]*)/,
  /^\s*pub\s+enum\s+([A-Za-z_][\w]*)/,
  /^\s*pub\s+trait\s+([A-Za-z_][\w]*)/,
  /^\s*impl(?:<[^>]*>)?\s+(?:[A-Za-z_][\w:<>, ]*\s+for\s+)?([A-Za-z_][\w]*)/,
];
const RB_PATTERNS = [
  /^\s*def\s+(?:self\.)?([A-Za-z_][\w?!]*)/,
  /^\s*class\s+([A-Za-z_][\w:]*)/,
  /^\s*module\s+([A-Za-z_][\w:]*)/,
];
// Java / Kotlin / C#: public types, and public members ONE level in (1-8
// leading spaces). Anything deeper is a local class or a lambda body.
const JVM_PATTERNS = [
  /^\s*(?:public|internal)\s+(?:static\s+|final\s+|abstract\s+|sealed\s+|partial\s+|data\s+|open\s+)*(?:class|interface|enum|record|object|struct)\s+([A-Za-z_][\w]*)/,
  /^\s{1,8}(?:public\s+)?(?:suspend\s+)?fun\s+([A-Za-z_][\w]*)/,
  /^\s{1,8}(?:public|internal)\s+(?:static\s+|final\s+|abstract\s+|open\s+|override\s+|virtual\s+|async\s+|suspend\s+)*(?:[\w.<>,?[\]]+\s+)?([A-Za-z_][\w]*)\s*\(/,
];
// The pre-1.3 union, kept for source extensions with no group of their own and
// for callers that do not say what language the text is.
const LEGACY_PATTERNS = [
  ...JS_PATTERNS,
  /^\s*(?:async\s+)?def\s+([A-Za-z_][\w]*)\s*\(/,
  /^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_][\w]*)\s*\(/,
  /^\s*pub\s+(?:async\s+)?fn\s+([A-Za-z_][\w]*)/,
];

const PATTERNS_BY_EXT = new Map([
  ['.js', JS_PATTERNS], ['.mjs', JS_PATTERNS], ['.cjs', JS_PATTERNS], ['.jsx', JS_PATTERNS],
  ['.ts', JS_PATTERNS], ['.tsx', JS_PATTERNS], ['.mts', JS_PATTERNS], ['.cts', JS_PATTERNS],
  ['.py', PY_PATTERNS], ['.pyi', PY_PATTERNS],
  ['.go', GO_PATTERNS],
  ['.rs', RS_PATTERNS],
  ['.rb', RB_PATTERNS],
  ['.java', JVM_PATTERNS], ['.kt', JVM_PATTERNS], ['.kts', JVM_PATTERNS], ['.cs', JVM_PATTERNS],
]);

const EXPORT_LIST_RE = /^\s*export\s*\{([^}]*)\}/;

function posix(p) {
  return String(p || '').replace(/\\/g, '/');
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

/**
 * The cache key for one repository. lessons.js keys its per-repo file on the
 * same value, so a brief and its lessons always describe the same checkout.
 * @param {object} wt worktree
 * @param {{repoKey?:string}} [opts]
 * @returns {string}
 */
export function repoKeyFor(wt, opts = {}) {
  return String(opts.repoKey || wt?.repoKey || wt?.repoPath || wt?.path || 'repo');
}

/**
 * Names a source file declares, capped at 20 per file.
 * The pattern set is chosen by extension so a JS `class Foo` stays out of the
 * list (only exports count there) while a Ruby or Python `class Foo` counts.
 * @param {string} text file contents
 * @param {string} [ext] lowercased extension with the dot, e.g. '.py'
 * @returns {string[]}
 */
export function parseExports(text, ext) {
  const patterns = PATTERNS_BY_EXT.get(String(ext || '').toLowerCase()) || LEGACY_PATTERNS;
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
    for (const re of patterns) {
      const m = re.exec(line);
      if (m && !names.includes(m[1])) {
        names.push(m[1]);
        break;
      }
    }
  }
  return names;
}

/**
 * Directories that hold this repo's tests, so a coder writing a new test knows
 * where it belongs without a listing turn.
 * @param {string[]} files POSIX relative paths
 * @returns {string[]} sorted, at most 20
 */
export function testDirsFrom(files) {
  const dirs = new Set();
  for (const file of files) {
    const segments = posix(file).split('/');
    const base = segments.pop() || '';
    for (let i = 0; i < segments.length; i += 1) {
      if (TEST_DIR_SEGMENTS.has(segments[i].toLowerCase())) dirs.add(segments.slice(0, i + 1).join('/'));
    }
    if (TEST_FILE_RE.test(base)) dirs.add(segments.join('/') || '.');
  }
  return [...dirs].sort().slice(0, TEST_DIR_CAP);
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
    const ext = extOf(f);
    const lang = ext ? (LANGUAGE_BY_EXT[ext] || ext.slice(1)) : '(none)';
    languages[lang] = (languages[lang] || 0) + 1;
  }

  const testDirs = testDirsFrom(kept);
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
    const names = parseExports(text, extOf(f));
    if (names.length > 0) exports[f] = names;
  }

  const brief = {
    sha: sha ?? null,
    generatedAt: new Date().toISOString(),
    tree,
    languages,
    testDirs,
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
 * Render the brief for a model prompt.
 *
 * The core (header, languages, commands, manifests, README head, tree, exports)
 * is always <= BRIEF_BYTE_CAP. The three optional attachments carry their own
 * budgets on top: instructions 6 KB, lessons 1.5 KB, retrieval whatever
 * KOLM_CODE_RETRIEVAL_BYTES allowed. Retrieval renders LAST so a consumer's own
 * byte cap eats snippets before it eats the file tree.
 *
 * @param {object} brief may carry `instructions`, `lessons`, `retrieval`
 * @returns {string}
 */
export function briefToPrompt(brief) {
  if (!brief) return '';
  // `head` is everything a reader needs before the file list; the optional
  // instructions block is spliced in right after it.
  const head = [];
  head.push(`REPO BRIEF (base ${brief.sha || 'unknown'})`);
  const langs = Object.entries(brief.languages || {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .map(([ext, n]) => `${ext} ${n}`)
    .join(', ');
  if (langs) head.push(`languages: ${langs}`);
  const cmd = brief.commands || {};
  head.push(`commands: test=${cmd.test || '-'} typecheck=${cmd.typecheck || '-'} lint=${cmd.lint || '-'} coverage=${cmd.coverage || '-'}`);
  for (const [name, m] of Object.entries(brief.manifests || {})) {
    if (name === 'package.json') {
      const scripts = Object.keys(m.scripts || {}).slice(0, 12).join(', ');
      head.push(`package.json: ${m.name || '?'}@${m.version || '?'} scripts[${scripts}] deps ${(m.dependencies || []).length}+${(m.devDependencies || []).length}`);
    } else {
      head.push(`${name}: ${JSON.stringify(m)}`);
    }
  }
  if (brief.readmeHead) {
    head.push('README (head):');
    head.push(brief.readmeHead.slice(0, 1200));
  }

  const tail = [];
  tail.push(`files (${(brief.tree || []).length} shown):`);
  tail.push((brief.tree || []).join('\n'));
  const exportEntries = Object.entries(brief.exports || {});
  if (exportEntries.length > 0) {
    tail.push('exports:');
    for (const [file, names] of exportEntries) tail.push(`  ${file}: ${names.join(', ')}`);
  }

  const headText = head.join('\n');
  let core = [...head, ...tail].join('\n');
  if (Buffer.byteLength(core, 'utf8') > BRIEF_BYTE_CAP) {
    const marker = '\n[brief truncated]';
    // -8 of slack: cutting mid-codepoint can re-encode as a 3-byte U+FFFD.
    const room = BRIEF_BYTE_CAP - Buffer.byteLength(marker, 'utf8') - 8;
    core = Buffer.from(core, 'utf8').subarray(0, room).toString('utf8') + marker;
  }

  const instructions = instructionsToPrompt(brief.instructions);
  const lessons = lessonsToPrompt(brief.lessons);
  const retrieval = retrievalToPrompt(brief.retrieval);
  // Fast path, and the byte-identity guarantee: no attachments, no change.
  if (!instructions && !lessons && !retrieval) return core;

  let text = core;
  if (instructions) {
    // The rules belong above the tree; when truncation ate the head (it cannot,
    // in practice, since the head is bounded well under 12 KB) fall back to
    // appending rather than splicing into the middle of a cut string.
    text = core.startsWith(headText)
      ? `${headText}\n${instructions}${core.slice(headText.length)}`
      : `${core}\n${instructions}`;
  }
  if (lessons) text += `\n${lessons}`;
  if (retrieval) text += `\n${retrieval}`;
  return text;
}

export default { repoBrief, briefToPrompt, repoKeyFor, parseExports, testDirsFrom, BRIEF_BYTE_CAP };
