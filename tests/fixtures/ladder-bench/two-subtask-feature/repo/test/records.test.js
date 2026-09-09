'use strict';
const test = require('node:test');
const assert = require('node:assert');
const { sampleRecords } = require('../src/records.js');

test('the sample set has three records', () => {
  const records = sampleRecords();
  assert.equal(records.length, 3);
  assert.equal(records[0].name, 'ada');
});
