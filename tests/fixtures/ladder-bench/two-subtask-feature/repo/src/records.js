'use strict';

// The one record shape every formatter renders: id, name, score.
function sampleRecords() {
  return [
    { id: 1, name: 'ada', score: 9 },
    { id: 2, name: 'linus', score: 7 },
    { id: 3, name: 'grace', score: 8 },
  ];
}

module.exports = { sampleRecords };
