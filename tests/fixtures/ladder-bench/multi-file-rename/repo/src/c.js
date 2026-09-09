'use strict';
const log = require('./log.js');

function traceStep(step) {
  return log.writeMsg('debug', 'step ' + step);
}

module.exports = { traceStep };
