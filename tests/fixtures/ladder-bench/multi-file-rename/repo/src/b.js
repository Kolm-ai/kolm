'use strict';
const { writeMsg } = require('./log.js');

function auditFailure(reason) {
  return writeMsg('error', 'failed ' + reason);
}

module.exports = { auditFailure };
