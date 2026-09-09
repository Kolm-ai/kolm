#!/usr/bin/env node
'use strict';
const { summarize } = require('../src/stats.js');

// Text output only. Every line is "<name> <value>".
function main(argv) {
  const s = summarize(argv);
  const lines = [
    'count ' + s.count,
    'sum ' + s.sum,
    'mean ' + s.mean,
    'min ' + s.min,
    'max ' + s.max,
  ];
  process.stdout.write(lines.join('\n') + '\n');
  return 0;
}

if (require.main === module) process.exitCode = main(process.argv.slice(2));

module.exports = { main };
