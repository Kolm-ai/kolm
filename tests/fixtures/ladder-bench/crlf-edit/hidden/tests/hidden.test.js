'use strict';
const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');
const { defaults, withDefaults } = require('../src/config.js');

const ROOT = path.join(__dirname, '..');

function assertCrlfOnly(rel) {
  const raw = fs.readFileSync(path.join(ROOT, rel), 'latin1');
  assert.ok(raw.indexOf('\r\n') !== -1, rel + ' has no CRLF line endings left');
  for (let i = 0; i < raw.length; i += 1) {
    if (raw[i] === '\n') {
      assert.equal(raw[i - 1], '\r', rel + ' has a bare LF at offset ' + i);
    }
  }
}

test('the timeout default is present', () => {
  assert.equal(defaults().timeoutMs, 30000);
  assert.equal(withDefaults({ timeoutMs: 5 }).timeoutMs, 5);
});

test('the other defaults are untouched', () => {
  assert.equal(defaults().retries, 3);
  assert.equal(defaults().backoffMs, 250);
  assert.equal(defaults().userAgent, 'bench-crlf/1.0');
});

test('every source file still uses CRLF', () => {
  assertCrlfOnly(path.join('src', 'config.js'));
  assertCrlfOnly(path.join('test', 'config.test.js'));
});
