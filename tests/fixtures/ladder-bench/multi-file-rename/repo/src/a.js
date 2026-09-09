'use strict';
const { writeMsg } = require('./log.js');

function auditLogin(user) {
  return writeMsg('info', 'login ' + user);
}

module.exports = { auditLogin };
