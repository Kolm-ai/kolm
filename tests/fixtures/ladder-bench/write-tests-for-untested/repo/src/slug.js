'use strict';

// A dependency-free slug builder. Written as a character loop rather than a
// regular expression so the ASCII-only rule is obvious at the call site.
function slugify(text) {
  const input = String(text === null || text === undefined ? '' : text).toLowerCase();
  let out = '';
  for (let i = 0; i < input.length; i += 1) {
    const ch = input[i];
    const isDigit = ch >= '0' && ch <= '9';
    const isAlpha = ch >= 'a' && ch <= 'z';
    if (isDigit || isAlpha) out += ch;
    else if (out.length > 0 && out[out.length - 1] !== '-') out += '-';
  }
  while (out.length > 0 && out[out.length - 1] === '-') out = out.slice(0, -1);
  return out;
}

module.exports = { slugify };
