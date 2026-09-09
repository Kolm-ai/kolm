'use strict';

// An in-memory log buffer. drain() hands the lines to the caller and clears
// the buffer, so a test never sees another test s lines.
const lines = [];

function writeMsg(level, text) {
  lines.push(String(level).toUpperCase() + ' ' + text);
  return lines[lines.length - 1];
}

function drain() {
  const out = lines.slice();
  lines.length = 0;
  return out;
}

module.exports = { writeMsg, drain };
