'use strict';
const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');
const log = require('../src/log.js');
const { auditLogin } = require('../src/a.js');
const { auditFailure } = require('../src/b.js');
const { traceStep } = require('../src/c.js');

const ROOT = path.join(__dirname, '..');
const FILES = ['src/log.js', 'src/a.js', 'src/b.js', 'src/c.js'];

test('the new name is exported and the old one is gone', () => {
  assert.equal(typeof log.writeMessage, 'function');
  assert.equal(log.writeMsg, undefined);
});

test('the old name appears in none of the four files', () => {
  for (const rel of FILES) {
    const raw = fs.readFileSync(path.join(ROOT, rel.split('/').join(path.sep)), 'utf8');
    assert.equal(raw.indexOf('writeMsg'), -1, rel + ' still contains writeMsg');
  }
});

test('behaviour is unchanged', () => {
  log.drain();
  assert.equal(auditLogin('ada'), 'INFO login ada');
  assert.equal(auditFailure('timeout'), 'ERROR failed timeout');
  assert.equal(traceStep('two'), 'DEBUG step two');
  assert.equal(log.drain().length, 3);
});
