'use strict';

// Formats a millisecond duration for the run log. Sub-second durations stay
// in milliseconds so short stages do not all render as "0.0s".
function formatDuration(ms) {
  if (!Number.isFinite(ms) || ms < 0) return "0ms";
  if (ms < 1000) return Math.round(ms) + "ms";
  if (ms < 60000) return (ms / 100).toFixed(1) + "s";
  const minutes = Math.floor(ms / 60000);
  const seconds = Math.round((ms % 60000) / 100);
  return minutes + "m " + seconds + "s";
}

module.exports = { formatDuration };
