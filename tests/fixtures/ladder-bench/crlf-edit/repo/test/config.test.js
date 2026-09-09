'use strict';
const test = require('node:test');
const assert = require('node:assert');
const { defaults, withDefaults } = require('../src/config.js');

test('the shipped defaults', () => {
  assert.equal(defaults().retries, 3);
  assert.equal(defaults().backoffMs, 250);
});

test('withDefaults overrides one field at a time', () => {
  assert.equal(withDefaults({ retries: 9 }).retries, 9);
  assert.equal(withDefaults({ retries: 9 }).backoffMs, 250);
});
