// kolm code — unified diff parser (owner C, shared with owner D).
//
// Small, dependency-free, and tolerant: it accepts `git diff` output (with the
// `diff --git` / `new file mode` / `rename from` headers and binary markers) and
// plain `diff -u` output (bare `---` / `+++` pairs). Hunk bodies are consumed by
// the counts declared in the `@@` header, so a removed line that itself starts
// with `---` (or an added line starting with `+++`) can never be mistaken for a
// file header.

/**
 * @typedef {{type:'+'|'-'|' ', text:string}} DiffLine
 * @typedef {{oldStart:number, oldLines:number, newStart:number, newLines:number, lines:DiffLine[]}} DiffHunk
 * @typedef {{path:string|null, oldPath:string|null, status:'A'|'M'|'D'|'R', binary:boolean,
 *            hunks:DiffHunk[], added:number, deleted:number}} DiffFile
 */

const HUNK_RE = /^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@/;
const DIFF_GIT_RE = /^diff --git (.+)$/;

/** Strip quoting and the a/ b/ prefix; '/dev/null' becomes null. */
function cleanPath(raw) {
  if (raw === undefined || raw === null) return null;
  let p = String(raw).trim();
  if (p.length === 0) return null;
  if (p.startsWith('"') && p.endsWith('"') && p.length >= 2) p = p.slice(1, -1);
  if (p === '/dev/null') return null;
  if (p.startsWith('a/') || p.startsWith('b/')) p = p.slice(2);
  return p;
}

/** Split the `diff --git a/x b/y` argument pair, tolerating spaces in names. */
function splitGitHeader(rest) {
  const trimmed = rest.trim();
  // Quoted form: "a/x y" "b/x y"
  const quoted = /^"(.*)"\s+"(.*)"$/.exec(trimmed);
  if (quoted) return [cleanPath(`"${quoted[1]}"`), cleanPath(`"${quoted[2]}"`)];
  const idx = trimmed.lastIndexOf(' b/');
  if (idx > 0) return [cleanPath(trimmed.slice(0, idx)), cleanPath(trimmed.slice(idx + 1))];
  const parts = trimmed.split(/\s+/);
  if (parts.length >= 2) return [cleanPath(parts[0]), cleanPath(parts[parts.length - 1])];
  return [null, null];
}

function newFile() {
  return { path: null, oldPath: null, status: 'M', binary: false, hunks: [], added: 0, deleted: 0 };
}

/**
 * Parse a unified diff into per-file records.
 * @param {string} text
 * @returns {DiffFile[]}
 */
export function parseUnifiedDiff(text) {
  const lines = String(text ?? '').split(/\r?\n/);
  /** @type {DiffFile[]} */
  const files = [];
  /** @type {DiffFile|null} */
  let file = null;
  /** @type {DiffHunk|null} */
  let hunk = null;
  let remOld = 0;
  let remNew = 0;

  const flush = () => {
    if (file) files.push(file);
    file = null;
    hunk = null;
    remOld = 0;
    remNew = 0;
  };
  const ensureFile = () => {
    if (!file) file = newFile();
    return file;
  };

  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];

    // Inside a hunk body: consume exactly the declared number of old/new lines.
    if (hunk && (remOld > 0 || remNew > 0)) {
      const c = line.charAt(0);
      if (c === '\\') continue; // "\ No newline at end of file"
      if (c === '+') {
        hunk.lines.push({ type: '+', text: line.slice(1) });
        file.added += 1;
        remNew -= 1;
        continue;
      }
      if (c === '-') {
        hunk.lines.push({ type: '-', text: line.slice(1) });
        file.deleted += 1;
        remOld -= 1;
        continue;
      }
      if (c === ' ' || line === '') {
        hunk.lines.push({ type: ' ', text: line.slice(1) });
        remOld -= 1;
        remNew -= 1;
        continue;
      }
      // Malformed body — fall through and re-read this line as a header.
      hunk = null;
      remOld = 0;
      remNew = 0;
    } else if (hunk) {
      hunk = null;
    }

    const gitHeader = DIFF_GIT_RE.exec(line);
    if (gitHeader) {
      flush();
      file = newFile();
      const [oldPath, newPath] = splitGitHeader(gitHeader[1]);
      file.oldPath = oldPath;
      file.path = newPath || oldPath;
      continue;
    }
    if (/^new file mode /.test(line)) {
      ensureFile().status = 'A';
      file.oldPath = null;
      continue;
    }
    if (/^deleted file mode /.test(line)) {
      ensureFile().status = 'D';
      continue;
    }
    if (/^rename from /.test(line)) {
      ensureFile().status = 'R';
      file.oldPath = cleanPath(line.slice('rename from '.length));
      continue;
    }
    if (/^rename to /.test(line)) {
      ensureFile().status = 'R';
      file.path = cleanPath(line.slice('rename to '.length));
      continue;
    }
    if (/^copy (from|to) /.test(line) || /^(old|new) mode /.test(line) || /^(similarity|dissimilarity) index /.test(line) || /^index /.test(line)) {
      ensureFile();
      continue;
    }
    if (/^GIT binary patch/.test(line) || /^Binary files? /.test(line)) {
      const f = ensureFile();
      f.binary = true;
      if (!f.path) {
        const m = /^Binary files? (.+) and (.+) differ/.exec(line);
        if (m) {
          f.oldPath = cleanPath(m[1]);
          f.path = cleanPath(m[2]) || f.oldPath;
        }
      }
      continue;
    }
    if (line.startsWith('--- ')) {
      const f = ensureFile();
      const p = cleanPath(line.slice(4));
      if (p === null) {
        f.status = 'A';
        f.oldPath = null;
      } else if (f.status !== 'R') {
        f.oldPath = f.oldPath || p;
      }
      continue;
    }
    if (line.startsWith('+++ ')) {
      const f = ensureFile();
      const p = cleanPath(line.slice(4));
      if (p === null) {
        f.status = 'D';
        if (!f.path) f.path = f.oldPath;
      } else if (f.status !== 'R') {
        f.path = f.path || p;
      }
      continue;
    }
    const h = HUNK_RE.exec(line);
    if (h) {
      const f = ensureFile();
      const oldLines = h[2] === undefined ? 1 : Number(h[2]);
      const newLines = h[4] === undefined ? 1 : Number(h[4]);
      hunk = {
        oldStart: Number(h[1]),
        oldLines,
        newStart: Number(h[3]),
        newLines,
        lines: [],
      };
      remOld = oldLines;
      remNew = newLines;
      f.hunks.push(hunk);
      continue;
    }
    // Anything else (commit headers, prose between files) is ignored.
  }
  flush();
  return files.filter((f) => f.path !== null || f.oldPath !== null);
}

/**
 * Summary numbers for the L0 size bands.
 * `patchLines` = added + deleted (the churn the reviewer has to read), not the
 * raw line count of the patch text.
 * @param {string} text
 * @returns {{files:number, added:number, deleted:number, patchLines:number, binaries:string[]}}
 */
export function diffStats(text) {
  const files = parseUnifiedDiff(text);
  let added = 0;
  let deleted = 0;
  const binaries = [];
  for (const f of files) {
    added += f.added;
    deleted += f.deleted;
    if (f.binary) binaries.push(f.path || f.oldPath);
  }
  return { files: files.length, added, deleted, patchLines: added + deleted, binaries };
}

export default { parseUnifiedDiff, diffStats };
