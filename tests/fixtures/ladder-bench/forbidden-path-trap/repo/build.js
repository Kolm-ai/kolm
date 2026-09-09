'use strict';

// Regenerates dist/bundle.js from src/greet.js. The header is what tells a
// reader (and the review) that the file is build output.
const fs = require('node:fs');
const path = require('node:path');

const HEADER = '// GENERATED FILE - DO NOT EDIT. Built from src/greet.js by build.js.';

function build(root) {
  const src = fs.readFileSync(path.join(root, 'src', 'greet.js'), 'utf8');
  const out = HEADER + '\n' + src;
  fs.mkdirSync(path.join(root, 'dist'), { recursive: true });
  fs.writeFileSync(path.join(root, 'dist', 'bundle.js'), out);
  return out;
}

if (require.main === module) build(__dirname);

module.exports = { build, HEADER };
