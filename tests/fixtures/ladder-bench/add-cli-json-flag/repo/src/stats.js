'use strict';

// Summarises a list of values. Non-numeric arguments are dropped rather than
// poisoning the whole summary with NaN.
function summarize(numbers) {
  const values = (numbers || []).map(Number).filter((n) => Number.isFinite(n));
  if (values.length === 0) return { count: 0, sum: 0, mean: 0, min: 0, max: 0 };
  const sum = values.reduce((a, b) => a + b, 0);
  return {
    count: values.length,
    sum,
    mean: sum / values.length,
    min: Math.min.apply(null, values),
    max: Math.max.apply(null, values),
  };
}

module.exports = { summarize };
