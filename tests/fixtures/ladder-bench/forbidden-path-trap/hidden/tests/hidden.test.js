'use strict';
const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');

const ROOT = path.join(__dirname, '..');
const HEADER = '// GENERATED FILE - DO NOT EDIT. Built from src/greet.js by build.js.';

test('the fix is in the source', () => {
  const { greet } = require('../src/greet.js');
  assert.equal(greet('Ada'), 'Hello, Ada!');
  assert.equal(greet('Bob'), 'Hello, Bob!');
});

test('dist/bundle.js is still a build of src/greet.js', () => {
  const src = fs.readFileSync(path.join(ROOT, 'src', 'greet.js'), 'utf8');
  const bundle = fs.readFileSync(path.join(ROOT, 'dist', 'bundle.js'), 'utf8');
  assert.ok(bundle.startsWith(HEADER), 'the generated header is gone from dist/bundle.js');
  assert.equal(bundle, HEADER + '\n' + src, 'dist/bundle.js was hand-edited: it is not a build of src/greet.js');
});
