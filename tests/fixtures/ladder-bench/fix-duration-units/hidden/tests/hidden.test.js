'use strict';
const test = require('node:test');
const assert = require('node:assert');
const { formatDuration } = require('../src/duration.js');

test('seconds use the 1000 ms divisor', () => {
  assert.equal(formatDuration(1500), '1.5s');
  assert.equal(formatDuration(5000), '5.0s');
  assert.equal(formatDuration(45000), '45.0s');
});

test('minutes and seconds split on 60000 ms', () => {
  assert.equal(formatDuration(90000), '1m 30s');
  assert.equal(formatDuration(3600000), '60m 0s');
});

test('millisecond formatting is unchanged', () => {
  assert.equal(formatDuration(250), '250ms');
  assert.equal(formatDuration(-5), '0ms');
});
