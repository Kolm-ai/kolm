'use strict';

// Defaults for the fetcher. Callers override single fields through
// withDefaults(), so every default lives in exactly one place.
function defaults() {
  return {
    retries: 3,
    backoffMs: 250,
    userAgent: 'bench-crlf/1.0',
  };
}

function withDefaults(options) {
  return Object.assign(defaults(), options || {});
}

module.exports = { defaults, withDefaults };
