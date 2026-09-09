'use strict';
const test = require('node:test');
const assert = require('node:assert');
const { formatDuration } = require('../src/duration.js');

test('sub-second durations render in milliseconds', () => {
  assert.equal(formatDuration(0), '0ms');
  assert.equal(formatDuration(250), '250ms');
  assert.equal(formatDuration(999), '999ms');
});

test('bad input falls back to zero', () => {
  assert.equal(formatDuration(-5), '0ms');
  assert.equal(formatDuration(NaN), '0ms');
});
