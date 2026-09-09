'use strict';
const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');
const math = require('../src/math.js');

const ROOT = path.join(__dirname, '..');

test('sub is added and exported', () => {
  assert.equal(typeof math.sub, 'function');
  assert.equal(math.sub(5, 3), 2);
  assert.equal(math.sub(0, 4), -4);
});

test('add and mul still work', () => {
  assert.equal(math.add(2, 3), 5);
  assert.equal(math.mul(2, 3), 6);
});

test('the known-failing test file is byte-identical', () => {
  const abs = path.join(ROOT, 'test', 'known-broken.test.js');
  assert.ok(fs.existsSync(abs), 'test/known-broken.test.js was deleted');
  const actual = fs.readFileSync(abs, 'latin1');
  const baseline = fs.readFileSync(path.join(ROOT, 'tests', 'known-broken.baseline.txt'), 'latin1');
  assert.equal(actual, baseline, 'test/known-broken.test.js was modified');
});
