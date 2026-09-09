'use strict';
const test = require('node:test');
const assert = require('node:assert');
const { greet } = require('../dist/bundle.js');

test('greet welcomes the caller by name', () => {
  assert.equal(greet('Ada'), 'Hello, Ada!');
});
