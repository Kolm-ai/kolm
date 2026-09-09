'use strict';
const test = require('node:test');
const assert = require('node:assert');
const { drain } = require('../src/log.js');
const { auditLogin } = require('../src/a.js');
const { auditFailure } = require('../src/b.js');
const { traceStep } = require('../src/c.js');

test('every caller writes one line', () => {
  drain();
  assert.equal(auditLogin('ada'), 'INFO login ada');
  assert.equal(auditFailure('timeout'), 'ERROR failed timeout');
  assert.equal(traceStep('two'), 'DEBUG step two');
  assert.equal(drain().length, 3);
});
