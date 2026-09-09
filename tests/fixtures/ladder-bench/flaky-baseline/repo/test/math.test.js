'use strict';
const test = require('node:test');
const assert = require('node:assert');
const { add, mul } = require('../src/math.js');

test('add and mul', () => {
  assert.equal(add(2, 3), 5);
  assert.equal(mul(2, 3), 6);
});
