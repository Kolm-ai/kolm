'use strict';
const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

const ROOT = path.join(__dirname, '..');

// Any *.test.js the coder added, anywhere but this hidden directory.
function findTestFiles(dir, found) {
  let entries = [];
  try {
    entries = fs.readdirSync(dir, { withFileTypes: true });
  } catch {
    return found;
  }
  for (const entry of entries) {
    const abs = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (entry.name === 'node_modules' || entry.name === '.git') continue;
      findTestFiles(abs, found);
    } else if (entry.name.endsWith('.test.js') && !entry.name.includes('hidden')) {
      found.push(abs);
    }
  }
  return found;
}

test('package.json runs node --test', () => {
  const raw = fs.readFileSync(path.join(ROOT, 'package.json'), 'utf8');
  const parsed = JSON.parse(raw);
  const script = String((parsed.scripts || {}).test || '');
  assert.ok(script.includes('node --test'), 'package.json test script is ' + JSON.stringify(script));
});

test('the added tests exist, name slugify and pass', () => {
  const files = findTestFiles(ROOT, []);
  assert.ok(files.length > 0, 'no *.test.js file was added');
  const mentions = files.some((f) => fs.readFileSync(f, 'utf8').includes('slugify'));
  assert.ok(mentions, 'no added test file mentions slugify');
  const res = spawnSync(process.execPath, ['--test', ...files], { cwd: ROOT, encoding: 'utf8' });
  assert.equal(res.status, 0, 'the added tests do not pass:\n' + String(res.stdout).slice(-2000));
});

test('slugify itself still behaves', () => {
  const { slugify } = require('../src/slug.js');
  assert.equal(slugify('Hello, World!'), 'hello-world');
  assert.equal(slugify('  spaced  out  '), 'spaced-out');
  assert.equal(slugify(null), '');
});
