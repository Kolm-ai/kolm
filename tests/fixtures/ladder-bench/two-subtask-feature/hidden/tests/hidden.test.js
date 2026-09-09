'use strict';
const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');
const { sampleRecords } = require('../src/records.js');

const ROOT = path.join(__dirname, '..');

test('toCsv renders the header and one line per record', () => {
  const { toCsv } = require('../src/csv.js');
  const csv = toCsv(sampleRecords());
  assert.equal(csv, ['id,name,score', '1,ada,9', '2,linus,7', '3,grace,8'].join('\n'));
});

test('toTable renders the header, a rule and one line per record', () => {
  const { toTable } = require('../src/table.js');
  const table = toTable(sampleRecords());
  assert.equal(table, ['id | name | score', '---', '1 | ada | 9', '2 | linus | 7', '3 | grace | 8'].join('\n'));
});

test('the formatters are independent modules', () => {
  const csvSource = fs.readFileSync(path.join(ROOT, 'src', 'csv.js'), 'utf8');
  const tableSource = fs.readFileSync(path.join(ROOT, 'src', 'table.js'), 'utf8');
  assert.equal(csvSource.indexOf('table.js'), -1, 'src/csv.js imports src/table.js');
  assert.equal(tableSource.indexOf('csv.js'), -1, 'src/table.js imports src/csv.js');
});

test('an integration test uses both formatters', () => {
  const abs = path.join(ROOT, 'test', 'integration.test.js');
  assert.ok(fs.existsSync(abs), 'test/integration.test.js was not created');
  const raw = fs.readFileSync(abs, 'utf8');
  assert.ok(raw.indexOf('csv') !== -1, 'the integration test does not use the CSV formatter');
  assert.ok(raw.indexOf('table') !== -1, 'the integration test does not use the table formatter');
});
