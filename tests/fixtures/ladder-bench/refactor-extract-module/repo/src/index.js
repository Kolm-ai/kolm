'use strict';

// Ranges are plain objects { start, end } with start <= end. Parsing is
// forgiving on whitespace because the values come from a config file.
function parseRange(text) {
  const parts = String(text).split('-');
  const start = Number(parts[0]);
  const end = Number(parts[1]);
  if (!Number.isFinite(start) || !Number.isFinite(end)) return { start: 0, end: 0 };
  return start <= end ? { start, end } : { start: end, end: start };
}

function clampRange(range, min, max) {
  return {
    start: Math.min(Math.max(range.start, min), max),
    end: Math.min(Math.max(range.end, min), max),
  };
}

function formatRange(range) {
  return range.start + '-' + range.end;
}

function rangeLength(range) {
  return range.end - range.start;
}

function containsValue(range, value) {
  return value >= range.start && value <= range.end;
}

function describeRange(range) {
  return formatRange(range) + ' (' + rangeLength(range) + ' wide)';
}

module.exports = {
  parseRange,
  clampRange,
  formatRange,
  rangeLength,
  containsValue,
  describeRange,
};
