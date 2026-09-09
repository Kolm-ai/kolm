// kolm code - task-focused retrieval (SOTA spec 1.1). Owner CTX.
//
// The coder's first turn is the expensive one: a small model that has to guess
// which of 400 files matters burns turns on read_file before it writes a line.
// This module does that search once, mechanically, with no model call: pull the
// identifiers and paths out of the task text, grep them across the tree, rank
// the files that carry them, and hand the coder the regions that actually hit.
//
// Everything here is deterministic (score desc, path asc) so a resumed run and
// a cached run render the same prompt bytes, and so a test can pin the ranking.
//
// This module owns the vendor-directory predicate and the ASCII coercion for
// the whole context pack: repo-brief.js imports them from here rather than the
// other way round, because repo-brief.js imports retrievalToPrompt and a cycle
// between the two would be fragile under ESM.

/** Directories whose contents are never the task's subject. */
export const VENDOR_DIRS = new Set([
  'node_modules', '.git', 'dist', 'build', 'out', 'coverage', 'vendor', 'target',
  '__pycache__', '.venv', 'venv', '.next', '.nuxt', '.cache', '.tox', '.mypy_cache',
  '.pytest_cache', '.gradle', 'Pods', 'bower_components', '.terraform',
]);

/** Extensions we never open: the bytes are not text a coder can read. */
const BINARY_EXT = new Set([
  '.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.ico', '.icns', '.svgz',
  '.pdf', '.zip', '.gz', '.tgz', '.bz2', '.xz', '.7z', '.rar', '.jar', '.war',
  '.wasm', '.so', '.dylib', '.dll', '.exe', '.bin', '.o', '.a', '.class',
  '.woff', '.woff2', '.ttf', '.otf', '.eot', '.mp3', '.mp4', '.mov', '.avi',
  '.wav', '.ogg', '.webm', '.sqlite', '.db', '.pyc', '.pack', '.idx',
]);

/** Generated dependency manifests: huge, and never the fix. */
const LOCKFILES = new Set([
  'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml', 'bun.lockb', 'npm-shrinkwrap.json',
  'poetry.lock', 'Pipfile.lock', 'Cargo.lock', 'go.sum', 'composer.lock', 'Gemfile.lock',
  'uv.lock', 'pdm.lock', 'deno.lock', 'flake.lock',
]);

/**
 * Words that appear in every task sentence. Applied only to bare lowercase
 * words, so `foo_bar`, `fooBar` and quoted spans survive the filter.
 */
const STOP_WORDS = new Set([
  'a', 'about', 'add', 'after', 'all', 'also', 'an', 'and', 'any', 'are', 'as', 'at',
  'back', 'be', 'because', 'been', 'before', 'both', 'but', 'by', 'call', 'called',
  'can', 'case', 'change', 'changes', 'class', 'code', 'current', 'currently', 'do',
  'does', 'done', 'each', 'else', 'error', 'every', 'file', 'files', 'fix', 'for',
  'from', 'function', 'get', 'give', 'goes', 'good', 'has', 'have', 'how', 'if',
  'in', 'instead', 'into', 'is', 'it', 'its', 'just', 'keep', 'like', 'make',
  'many', 'method', 'more', 'most', 'must', 'need', 'needs', 'never', 'new', 'no',
  'not', 'now', 'of', 'on', 'one', 'only', 'or', 'other', 'our', 'out', 'over',
  'own', 'please', 'put', 'repo', 'repository', 'return', 'returns', 'run', 'same',
  'see', 'set', 'should', 'so', 'some', 'still', 'such', 'take', 'test', 'tests',
  'than', 'that', 'the', 'their', 'them', 'then', 'there', 'these', 'they', 'this',
  'those', 'through', 'to', 'too', 'two', 'up', 'update', 'updates', 'use', 'used',
  'using', 'value', 'very', 'want', 'was', 'way', 'we', 'were', 'what', 'when',
  'where', 'which', 'while', 'why', 'will', 'with', 'without', 'work', 'works',
  'would', 'you', 'your',
]);

/** A line that declares something, in any of the languages the brief parses. */
const DEF_LINE_RE = new RegExp(
  '^\\s*(?:@\\w+\\s+)?(?:export\\s+|default\\s+|pub(?:\\([^)]*\\))?\\s+|public\\s+|private\\s+|protected\\s+|internal\\s+|static\\s+|final\\s+|abstract\\s+|open\\s+|override\\s+|async\\s+|declare\\s+)*'
  + '(?:function|class|const|let|var|def|func|fn|struct|enum|impl|trait|type|interface|module|record|object)\\b',
);

/** Paths that look like a test for something else. */
const TEST_PATH_RE = new RegExp(
  '(?:^|/)(?:tests?|__tests__|spec|specs|testing)/|'
  + '(?:^|/)(?:test_[^/]+|[^/]+_test)\\.[A-Za-z0-9]+$|'
  + '\\.(?:test|spec)\\.[A-Za-z0-9]+$|'
  + '(?:^|/)[A-Za-z0-9_.-]*Test(?:s)?\\.[A-Za-z0-9]+$',
);

export const DEFAULT_MAX_FILES = 12;
export const DEFAULT_MAX_BYTES = 16 * 1024;
const MAX_TERMS = 14;
const MAX_SCAN_FILES = 1200;
const MAX_FILE_BYTES = 256 * 1024;
const CONTEXT_LINES = 12;
const WHOLE_FILE_LINES = 60;
const REGIONS_PER_FILE = 3;
const HITS_PER_TERM_PER_FILE = 5;
const SEARCH_LIMIT = 200;
const MIN_SNIPPET_ROOM = 200;
const NAMED_PATH_BONUS = 12;

const PROMPT_HEADER =
  'RELEVANT CODE (found by searching the task terms; read the full files before editing)';

/** Slash direction is a host detail; every path in the pack is POSIX. */
export function posixPath(p) {
  return String(p || '').replace(/\\/g, '/');
}

/**
 * Lowercased extension of a path, dot included ('' when there is none).
 * @param {string} file
 * @returns {string}
 */
export function extOf(file) {
  const base = posixPath(file).split('/').pop() || '';
  const dot = base.lastIndexOf('.');
  return dot > 0 ? base.slice(dot).toLowerCase() : '';
}

/**
 * True when any path segment is a vendored / generated directory.
 * repo-brief.js uses this too, so both halves of the pack skip the same tree.
 * @param {string} file
 * @returns {boolean}
 */
export function isVendored(file) {
  return posixPath(file).split('/').some((seg) => VENDOR_DIRS.has(seg));
}

/**
 * Coerce text to ASCII so it can reach a prompt or a terminal unchanged.
 * Smart quotes, dashes and ellipses become their ASCII shape; anything else
 * outside 0x20-0x7E (tabs and newlines excepted) is dropped.
 * @param {string} text
 * @returns {string}
 */
export function toAscii(text) {
  return String(text ?? '')
    .replace(/[\u2018\u2019\u201A\u2032]/g, "'")
    .replace(/[\u201C\u201D\u201E\u2033]/g, '"')
    .replace(/[\u2010-\u2015\u2212]/g, '-')
    .replace(/\u2026/g, '...')
    .replace(/[\u00A0\u2000-\u200B\u202F\u205F\u3000]/g, ' ')
    .replace(/\r\n/g, '\n')
    .replace(/[^\x09\x0A\x20-\x7E]/g, '');
}

/** Is this a file we are willing to open and rank? */
function isRetrievable(file) {
  const p = posixPath(file);
  if (!p || isVendored(p)) return false;
  const base = p.split('/').pop() || '';
  if (LOCKFILES.has(base)) return false;
  if (BINARY_EXT.has(extOf(p))) return false;
  if (base.startsWith('.') && !base.startsWith('.env')) {
    // Dotfiles are config, not the task's subject - except the ones a task
    // usually names outright, which the path bonus can still surface.
    return /^\.(?:eslintrc|babelrc|cursorrules|prettierrc)/.test(base);
  }
  return true;
}

/** A bare lowercase word carries no structure, so the stop list applies. */
function isBareWord(term) {
  return /^[a-z][a-z0-9]*$/.test(term);
}

function pushTerm(map, raw, weight) {
  const term = String(raw || '').trim();
  if (term.length < 3 || term.length > 120) return;
  if (isBareWord(term) && STOP_WORDS.has(term)) return;
  const key = term.toLowerCase();
  const existing = map.get(key);
  // A structured mention outranks the same word seen as prose.
  if (existing && existing.weight >= weight) return;
  map.set(key, { term, weight });
}

/**
 * Pull the identifiers, paths and quoted spans a task names out of its text.
 * Structured mentions (backticks, quotes, paths, `foo(`) weigh 1; bare prose
 * words weigh 0.6, so "the duration parser" still finds duration.js without
 * letting prose outvote an explicit identifier.
 * @param {{title?:string, text?:string}|string} task
 * @returns {Array<{term:string, weight:number}>} deterministic, at most 14
 */
export function extractTerms(task) {
  const title = typeof task === 'string' ? '' : String(task?.title ?? '');
  const body = typeof task === 'string' ? task : String(task?.text ?? '');
  const source = `${title}\n${body}`;
  const terms = new Map();

  for (const m of source.matchAll(/`([^`\n]{2,120})`/g)) pushTerm(terms, m[1], 1);
  for (const m of source.matchAll(/"([^"\n]{2,120})"/g)) pushTerm(terms, m[1], 1);
  for (const m of source.matchAll(/'([^'\n]{2,120})'/g)) pushTerm(terms, m[1], 1);
  // Paths: at least one slash, or a bare filename with a source-ish extension.
  for (const m of source.matchAll(/\b[\w.@-]+(?:\/[\w.@-]+)+\b/g)) pushTerm(terms, m[0], 1);
  for (const m of source.matchAll(/\b[\w-]+\.[A-Za-z0-9]{1,8}\b/g)) pushTerm(terms, m[0], 1);
  // Call shapes: the name in front of an open paren.
  for (const m of source.matchAll(/\b([A-Za-z_$][\w$]{2,})\s*\(/g)) pushTerm(terms, m[1], 1);
  // CamelCase / lowerCamelCase / snake_case / SCREAMING_SNAKE identifiers.
  for (const m of source.matchAll(/\b[A-Za-z][a-z0-9]*(?:[A-Z][a-z0-9]*)+\b/g)) pushTerm(terms, m[0], 1);
  for (const m of source.matchAll(/\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+\b/g)) pushTerm(terms, m[0], 1);
  // Prose, last and weakest.
  for (const m of source.matchAll(/\b[A-Za-z][A-Za-z0-9]{2,}\b/g)) pushTerm(terms, m[0].toLowerCase(), 0.6);

  const out = [];
  for (const value of terms.values()) {
    out.push(value);
    if (out.length >= MAX_TERMS) break;
  }
  return out;
}

/**
 * A search pattern that means the same thing to a JS RegExp (fake driver) and
 * to `git grep`'s basic regex (local driver): every letter becomes a two-way
 * bracket class, the dot is escaped, everything else is already literal in
 * both dialects. Terms with any other character are matched on paths only.
 * @param {string} term
 * @returns {string|null}
 */
export function searchPatternFor(term) {
  if (!/^[A-Za-z0-9_./-]{3,}$/.test(term)) return null;
  let out = '';
  for (const ch of term) {
    if (/[A-Za-z]/.test(ch)) out += `[${ch.toUpperCase()}${ch.toLowerCase()}]`;
    else if (ch === '.') out += '\\.';
    else out += ch;
  }
  return out;
}

/** Does the task name this exact path (or its basename as a path token)? */
function namesPathOutright(file, terms) {
  const p = posixPath(file).toLowerCase();
  const base = p.split('/').pop();
  for (const { term } of terms) {
    const t = posixPath(term).toLowerCase().replace(/^\.\//, '');
    if (!t.includes('/') && !t.includes('.')) continue;
    if (t === p || t === base) return true;
    if (p.endsWith(`/${t}`)) return true;
  }
  return false;
}

async function listCandidates(wt) {
  let files = [];
  try {
    // Drivers expose listFiles({glob, limit}); `glob` is the spec's name for
    // the same capability, so accept either shape.
    if (typeof wt?.listFiles === 'function') files = (await wt.listFiles({ limit: 5000 })) || [];
    else if (typeof wt?.glob === 'function') files = (await wt.glob('**/*', { limit: 5000 })) || [];
  } catch {
    files = [];
  }
  return files.map(posixPath).filter(isRetrievable).sort().slice(0, MAX_SCAN_FILES);
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
 * Hits per file, either from the driver's grep or (when it has none) by
 * reading the candidate set. Returns Map<path, Array<{line, text, weight}>>.
 */
async function collectHits(wt, candidates, terms) {
  const allowed = new Set(candidates);
  const hits = new Map();
  const seen = new Set(); // path:line:term - one term counts a line once
  const add = (path, line, text, weight, termKey) => {
    if (!allowed.has(path)) return;
    const dedupe = `${path}:${line}:${termKey}`;
    if (seen.has(dedupe)) return;
    seen.add(dedupe);
    if (!hits.has(path)) hits.set(path, []);
    hits.get(path).push({ line, text: String(text).slice(0, 400), weight });
  };

  const searchable = [];
  for (const t of terms) {
    const pattern = searchPatternFor(t.term);
    if (pattern) searchable.push({ ...t, pattern });
  }

  if (typeof wt?.search === 'function' && searchable.length > 0) {
    for (const t of searchable) {
      let found = [];
      try {
        found = (await wt.search(t.pattern, { limit: SEARCH_LIMIT })) || [];
      } catch {
        found = [];
      }
      const perFile = new Map();
      for (const hit of found) {
        const p = posixPath(hit?.path);
        const n = perFile.get(p) || 0;
        if (n >= HITS_PER_TERM_PER_FILE) continue;
        perFile.set(p, n + 1);
        add(p, Number(hit?.line) || 1, hit?.text ?? '', t.weight, t.term.toLowerCase());
      }
    }
    return hits;
  }

  // Fallback: no grep on this driver, so read the candidates once each.
  const regexes = searchable.map((t) => ({ ...t, re: new RegExp(t.pattern) }));
  for (const path of candidates) {
    const text = await readMaybe(wt, path);
    if (text === null || text.length > MAX_FILE_BYTES) continue;
    const lines = text.split(/\r?\n/);
    for (const t of regexes) {
      let n = 0;
      for (let i = 0; i < lines.length && n < HITS_PER_TERM_PER_FILE; i += 1) {
        if (!t.re.test(lines[i])) continue;
        n += 1;
        add(path, i + 1, lines[i], t.weight, t.term.toLowerCase());
      }
    }
  }
  return hits;
}

function scoreFile(path, fileHits, terms) {
  const lower = posixPath(path).toLowerCase();
  let pathWeight = 0;
  const pathTerms = [];
  for (const { term, weight } of terms) {
    if (lower.includes(term.toLowerCase())) {
      pathWeight += weight;
      pathTerms.push(term);
    }
  }
  let defHits = 0;
  let bodyHits = 0;
  for (const hit of fileHits || []) {
    if (DEF_LINE_RE.test(hit.text)) defHits += hit.weight;
    else bodyHits += hit.weight;
  }
  const named = namesPathOutright(path, terms);
  const score = (3 * pathWeight) + (2 * defHits) + bodyHits + (named ? NAMED_PATH_BONUS : 0);

  const why = [];
  if (named) why.push('named in the task');
  if (pathTerms.length > 0) why.push(`path matches ${pathTerms.slice(0, 3).join(', ')}`);
  if (defHits > 0) why.push(`${Math.round(defHits * 10) / 10} definition hit(s)`);
  if (bodyHits > 0) why.push(`${Math.round(bodyHits * 10) / 10} body hit(s)`);
  return { score: Math.round(score * 100) / 100, why: why.join('; ') || 'no direct term hit' };
}

function mergeRegions(regions) {
  const sorted = regions.slice().sort((a, b) => a.start - b.start || a.end - b.end);
  const out = [];
  for (const r of sorted) {
    const last = out[out.length - 1];
    if (last && r.start <= last.end + 1) last.end = Math.max(last.end, r.end);
    else out.push({ start: r.start, end: r.end });
  }
  return out;
}

/** Regions of `lines` worth showing for one file, given its hits. */
function regionsFor(lines, fileHits) {
  if (lines.length <= WHOLE_FILE_LINES) return [{ start: 1, end: lines.length || 1 }];
  const sorted = (fileHits || []).slice().sort((a, b) => a.line - b.line);
  const defs = sorted.filter((h) => DEF_LINE_RE.test(h.text));
  const anchors = [];
  for (const h of defs) {
    if (anchors.length >= REGIONS_PER_FILE) break;
    anchors.push(h.line);
  }
  for (const h of sorted) {
    if (anchors.length >= REGIONS_PER_FILE) break;
    if (!anchors.includes(h.line)) anchors.push(h.line);
  }
  if (anchors.length === 0) return [{ start: 1, end: Math.min(WHOLE_FILE_LINES, lines.length) }];
  return mergeRegions(anchors.map((line) => ({
    start: Math.max(1, line - CONTEXT_LINES),
    end: Math.min(lines.length, line + CONTEXT_LINES),
  }))).slice(0, REGIONS_PER_FILE);
}

function snippetCost(path, region, text) {
  // What the rendered prompt will charge for this snippet: header + body.
  return Buffer.byteLength(`${path}:${region.start}-${region.end}\n${text}\n\n`, 'utf8');
}

/**
 * Find the code a task is about, without a model call.
 *
 * @param {object} wt worktree (listFiles/readFile required; search optional)
 * @param {{title?:string, text?:string}|string} task
 * @param {{maxFiles?:number, maxBytes?:number, config?:object}} [opts]
 * @returns {Promise<{terms:string[],
 *   files:Array<{path:string, score:number, why:string}>,
 *   snippets:Array<{path:string, startLine:number, endLine:number, text:string, score:number}>,
 *   bytes:number, maxBytes:number}>}
 */
export async function retrieveContext(wt, task, opts = {}) {
  const maxFiles = Number(opts.maxFiles) > 0 ? Math.trunc(opts.maxFiles) : DEFAULT_MAX_FILES;
  const configured = Number(opts.maxBytes ?? opts.config?.retrievalBytes);
  const maxBytes = Number.isFinite(configured) && configured > 0 ? Math.trunc(configured) : DEFAULT_MAX_BYTES;
  const terms = extractTerms(task);
  const empty = { terms: terms.map((t) => t.term), files: [], snippets: [], bytes: 0, maxBytes };
  if (!wt || terms.length === 0) return empty;

  const candidates = await listCandidates(wt);
  if (candidates.length === 0) return empty;

  const hits = await collectHits(wt, candidates, terms);
  const scored = [];
  for (const path of candidates) {
    const fileHits = hits.get(path);
    const { score, why } = scoreFile(path, fileHits, terms);
    if (score <= 0) continue;
    scored.push({ path, score, why });
  }
  scored.sort((a, b) => b.score - a.score || a.path.localeCompare(b.path));
  const files = scored.slice(0, maxFiles);
  if (files.length === 0) return empty;

  // Budget: the rendered block must fit maxBytes, headers included.
  let room = maxBytes - Buffer.byteLength(`${PROMPT_HEADER}\nterms: ${terms.map((t) => t.term).join(', ')}\n\n`, 'utf8');
  const snippets = [];
  let bytes = 0;
  const emitted = new Set();

  const emit = (path, region, lines, score) => {
    if (room < MIN_SNIPPET_ROOM) return false;
    const key = `${path}:${region.start}-${region.end}`;
    if (emitted.has(key)) return true;
    const text = toAscii(lines.slice(region.start - 1, region.end).join('\n'));
    const cost = snippetCost(path, region, text);
    if (cost > room) return true; // a smaller snippet may still fit
    emitted.add(key);
    room -= cost;
    bytes += Buffer.byteLength(text, 'utf8');
    snippets.push({ path, startLine: region.start, endLine: region.end, text, score });
    return true;
  };

  const bodies = new Map();
  for (const f of files) {
    const text = await readMaybe(wt, f.path);
    if (text === null || text.length > MAX_FILE_BYTES) continue;
    const lines = toAscii(text).split('\n');
    bodies.set(f.path, lines);
    for (const region of regionsFor(lines, hits.get(f.path))) {
      if (!emit(f.path, region, lines, f.score)) break;
    }
    if (room < MIN_SNIPPET_ROOM) break;
  }

  // The coder needs the existing test style for whatever it is about to change,
  // so a test file whose path names a top file's basename rides along.
  if (room >= MIN_SNIPPET_ROOM) {
    const bases = new Set();
    for (const f of files) {
      const base = f.path.split('/').pop().replace(/\.[A-Za-z0-9]+$/, '').toLowerCase();
      if (base.length >= 3) bases.add(base);
    }
    const testFiles = candidates.filter((p) => TEST_PATH_RE.test(p) && !bodies.has(p));
    for (const path of testFiles) {
      if (room < MIN_SNIPPET_ROOM) break;
      const lower = path.toLowerCase();
      let matched = null;
      for (const base of bases) {
        if (lower.includes(base)) { matched = base; break; }
      }
      if (!matched) continue;
      const text = await readMaybe(wt, path);
      if (text === null || text.length > MAX_FILE_BYTES) continue;
      const lines = toAscii(text).split('\n');
      const region = { start: 1, end: Math.min(WHOLE_FILE_LINES, lines.length || 1) };
      emit(path, region, lines, 0);
    }
  }

  return { terms: terms.map((t) => t.term), files, snippets, bytes, maxBytes };
}

/**
 * Render a retrieval for a model prompt. ASCII, and never wider than the
 * retrieval's own byte budget.
 * @param {{terms?:string[], files?:Array<object>, snippets?:Array<object>, maxBytes?:number}} retrieval
 * @returns {string}
 */
export function retrievalToPrompt(retrieval) {
  if (!retrieval) return '';
  const files = Array.isArray(retrieval.files) ? retrieval.files : [];
  const snippets = Array.isArray(retrieval.snippets) ? retrieval.snippets : [];
  if (files.length === 0 && snippets.length === 0) return '';

  const cap = Number(retrieval.maxBytes) > 0 ? Math.trunc(retrieval.maxBytes) : DEFAULT_MAX_BYTES;
  const out = [PROMPT_HEADER];
  const terms = Array.isArray(retrieval.terms) ? retrieval.terms : [];
  if (terms.length > 0) out.push(`terms: ${toAscii(terms.join(', '))}`);
  if (snippets.length === 0) out.push(`files: ${files.map((f) => f.path).join(', ')}`);
  for (const s of snippets) {
    out.push('');
    out.push(`${s.path}:${s.startLine}-${s.endLine}`);
    out.push(toAscii(s.text));
  }
  let text = out.join('\n');
  if (Buffer.byteLength(text, 'utf8') > cap) {
    const marker = '\n[retrieval truncated]';
    // -8 of slack: cutting mid-codepoint can re-encode as a 3-byte U+FFFD.
    const roomBytes = cap - Buffer.byteLength(marker, 'utf8') - 8;
    text = Buffer.from(text, 'utf8').subarray(0, Math.max(0, roomBytes)).toString('utf8') + marker;
  }
  return text;
}

export default { retrieveContext, retrievalToPrompt, extractTerms, isVendored, toAscii };
