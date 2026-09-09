'use strict';

// Plain float arithmetic; the callers round for display.
function add(a, b) {
  return a + b;
}

function mul(a, b) {
  return a * b;
}

module.exports = { add, mul };
