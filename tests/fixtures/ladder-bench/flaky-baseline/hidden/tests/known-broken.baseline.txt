'use strict';
// KNOWN FAILURE: tracked as issue #42. The rounding difference lives in the
// upstream service, not here. Do not change this file.
const test = require('node:test');
const assert = require('node:assert');
const { add } = require('../src/math.js');

test('rounding matches the legacy service (known failure, issue #42)', () => {
  assert.equal(add(0.1, 0.2), 0.3);
});
