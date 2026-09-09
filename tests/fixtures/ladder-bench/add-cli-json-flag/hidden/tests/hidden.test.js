'use strict';
const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

const ROOT = path.join(__dirname, '..');

function runCli(args) {
  return spawnSync(process.execPath, [path.join(ROOT, 'bin', 'cli.js'), ...args], {
    cwd: ROOT,
    encoding: 'utf8',
  });
}

test('--json prints one line of JSON', () => {
  const res = runCli(['--json', '1', '2', '3']);
  assert.equal(res.status, 0);
  const lines = String(res.stdout).split('\n').filter((l) => l.trim() !== '');
  assert.equal(lines.length, 1);
  const parsed = JSON.parse(lines[0]);
  assert.equal(parsed.count, 3);
  assert.equal(parsed.sum, 6);
  assert.equal(parsed.mean, 2);
  assert.equal(parsed.min, 1);
  assert.equal(parsed.max, 3);
});

test('the text output is unchanged', () => {
  const res = runCli(['1', '2', '3']);
  assert.equal(res.status, 0);
  assert.equal(String(res.stdout), ['count 3', 'sum 6', 'mean 2', 'min 1', 'max 3'].join('\n') + '\n');
});

test('the README documents the flag', () => {
  const readme = fs.readFileSync(path.join(ROOT, 'README.md'), 'utf8');
  assert.ok(readme.includes('--json'), 'README.md does not mention --json');
});
