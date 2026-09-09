'use strict';
const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');

const ROOT = path.join(__dirname, '..');
const MOVED = ['parseRange', 'clampRange', 'formatRange'];
const ALL = MOVED.concat(['rangeLength', 'containsValue', 'describeRange']);

test('src/range.js exists and defines the three moved functions', () => {
  const abs = path.join(ROOT, 'src', 'range.js');
  assert.ok(fs.existsSync(abs), 'src/range.js was not created');
  const range = require('../src/range.js');
  for (const name of MOVED) {
    assert.equal(typeof range[name], 'function', 'src/range.js does not export ' + name);
  }
});

test('src/index.js still exports all six names', () => {
  const api = require('../src/index.js');
  for (const name of ALL) {
    assert.equal(typeof api[name], 'function', 'src/index.js no longer exports ' + name);
  }
});

test('src/index.js no longer defines the moved functions', () => {
  const raw = fs.readFileSync(path.join(ROOT, 'src', 'index.js'), 'utf8');
  for (const name of MOVED) {
    assert.equal(raw.indexOf('function ' + name), -1, 'src/index.js still defines ' + name);
  }
});

test('behaviour is unchanged through either module', () => {
  const api = require('../src/index.js');
  const range = require('../src/range.js');
  assert.deepEqual(api.parseRange('3-9'), { start: 3, end: 9 });
  assert.deepEqual(range.parseRange('9-3'), { start: 3, end: 9 });
  assert.equal(api.formatRange(range.clampRange({ start: 3, end: 9 }, 0, 5)), '3-5');
  assert.equal(api.describeRange({ start: 2, end: 6 }), '2-6 (4 wide)');
});
