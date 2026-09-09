'use strict';
const test = require('node:test');
const assert = require('node:assert');
const { summarize } = require('../src/stats.js');

test('summarize reports the five fields', () => {
  assert.deepEqual(summarize(['1', '2', '3']), { count: 3, sum: 6, mean: 2, min: 1, max: 3 });
});

test('an empty list summarises to zeroes', () => {
  assert.deepEqual(summarize([]), { count: 0, sum: 0, mean: 0, min: 0, max: 0 });
});
