'use strict';
const test = require('node:test');
const assert = require('node:assert');
const api = require('../src/index.js');

test('parse, clamp and format round-trip', () => {
  const range = api.parseRange('3-9');
  assert.deepEqual(range, { start: 3, end: 9 });
  assert.equal(api.formatRange(api.clampRange(range, 0, 5)), '3-5');
});

test('the derived helpers still work', () => {
  const range = api.parseRange('2-6');
  assert.equal(api.rangeLength(range), 4);
  assert.equal(api.containsValue(range, 2), true);
  assert.equal(api.containsValue(range, 7), false);
  assert.equal(api.describeRange(range), '2-6 (4 wide)');
});
