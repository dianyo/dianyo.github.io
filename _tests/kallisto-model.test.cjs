const test = require('node:test');
const assert = require('node:assert/strict');
const model = require('../assets/kallisto/model.js');

const equalLengths = [1, 1, 1];
const informativeClasses = [
  { members: [0], count: 100 },
  { members: [1], count: 10 },
  { members: [0, 1], count: 90 }
];
function close(actual, expected, tolerance = 1e-10) {
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} should equal ${expected}`);
}
function vectorClose(actual, expected, tolerance = 1e-10) {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, i) => close(value, expected[i], tolerance));
}

test('k-mers retain every zero-based position and index deduplicates transcript membership', () => {
  assert.deepEqual(model.kmers(' actga ', 3), [
    { kmer: 'ACT', position: 0 }, { kmer: 'CTG', position: 1 }, { kmer: 'TGA', position: 2 }
  ]);
  const index = model.buildIndex(model.TRANSCRIPTS, 3);
  assert.deepEqual(index.get('ACT'), [0, 1]);
  assert.deepEqual(index.get('TGA'), [0, 1, 2]);
  assert.deepEqual(index.get('ACG'), [0, 2]);
  assert.deepEqual(index.get('CCT'), [1]);
  const repeat = model.buildIndex([{ sequence: 'AAAAA' }, { sequence: 'AAAA' }], 3);
  assert.deepEqual(repeat.get('AAA'), [0, 1]);
  assert.throws(() => model.kmers('ACGT', 0), /positive integer/);
});

test('pseudoalignment intersects known k-mers and skips unknown k-mers', () => {
  const index = model.buildIndex(model.TRANSCRIPTS, 3);
  const result = model.pseudoalign('actgacgta', 3, index);
  assert.equal(result.status, 'matched');
  assert.deepEqual(result.compatible, [0]);
  assert.equal(result.matchedCount, 7);
  assert.equal(result.totalKmers, 7);
  assert.deepEqual(result.steps.map(step => step.remaining), [
    [0, 1], [0, 1], [0, 1], [0, 1], [0], [0], [0]
  ]);
  const withUnknown = model.pseudoalign('AAACT', 3, index);
  assert.equal(withUnknown.status, 'matched');
  assert.deepEqual(withUnknown.compatible, [0, 1]);
  assert.deepEqual(withUnknown.steps.map(step => step.matched), [false, false, true]);
});

test('no hits, invalid sequences, short reads and contradictory evidence stay unassigned', () => {
  const index = model.buildIndex(model.TRANSCRIPTS, 3);
  for (const [read, status] of [['AAAAA', 'no-hits'], ['AC', 'too-short'], ['ACN', 'invalid'], ['', 'invalid']]) {
    const result = model.pseudoalign(read, 3, index);
    assert.equal(result.status, status);
    assert.deepEqual(result.compatible, []);
  }
  assert.equal(model.pseudoalign('ACT', 0, index).status, 'invalid');
  // GGT supports T3; CCT supports T2. The final TGA supports all transcripts,
  // but must not resurrect an intersection already emptied by contradiction.
  const conflict = model.pseudoalign('GGTCCTGA', 3, index);
  assert.equal(conflict.status, 'conflict');
  assert.deepEqual(conflict.compatible, []);
  assert.deepEqual(conflict.steps.at(-1).compatible, [0, 1, 2]);
  assert.deepEqual(conflict.steps.at(-1).remaining, []);
});

test('aggregation merges identical compatibility rows and conserves input counts', () => {
  const index = model.buildIndex(model.TRANSCRIPTS, 3);
  const result = model.aggregate([
    { sequence: 'ACT', count: 2 }, { sequence: 'CTG', count: 3 },
    { sequence: 'CCT', count: 4 }, { sequence: 'AAAA', count: 5 },
    { sequence: 'AC', count: 1 }, { sequence: 'N', count: 2 }
  ], 3, index);
  assert.deepEqual(result.classes, [{ members: [0, 1], count: 5 }, { members: [1], count: 4 }]);
  assert.equal(result.assigned, 9);
  assert.equal(result.unassigned, 8);
  assert.equal(result.assigned + result.unassigned, 17);
  assert.equal(result.results.length, 6);
  assert.throws(() => model.aggregate([{ sequence: 'ACT', count: -1 }], 3, index), /nonnegative integers/);
});

test('compressed classes preserve exactly the read-level log likelihood in this model', () => {
  const alpha = [0.5, 0.3, 0.2];
  const lengths = [5, 8, 6];
  const individualLikelihood = 3 * Math.log(0.5 / 5 + 0.3 / 8) + Math.log(0.2 / 6);
  const classes = [{ members: [0, 1], count: 3 }, { members: [2], count: 1 }];
  close(model.likelihood(classes, alpha, lengths), individualLikelihood);
});

test('E-step allocations and two hand-calculated EM steps agree', () => {
  const allocation = model.expectation(informativeClasses, [0.5, 0.5, 0], equalLengths);
  vectorClose(allocation.allocations[2].weights, [0.5, 0.5, 0]);
  vectorClose(allocation.allocations[2].expected, [45, 45, 0]);
  vectorClose(allocation.counts, [145, 55, 0]);
  const first = model.emStep(informativeClasses, [0.5, 0.5, 0], equalLengths);
  vectorClose(first.alpha, [0.725, 0.275, 0]);
  close(first.delta, 0.225);
  const second = model.emStep(informativeClasses, first.alpha, equalLengths);
  vectorClose(second.alpha, [0.82625, 0.17375, 0]);
  vectorClose(second.counts, [165.25, 34.75, 0]);
});

test('unequal effective lengths influence ambiguous allocations', () => {
  const result = model.expectation([{ members: [0, 1], count: 90 }], [0.5, 0.5, 0], [1, 2, 1]);
  vectorClose(result.counts, [60, 30, 0]);
});

test('EM increases likelihood, conserves counts, and reaches the analytic equal-length solution', () => {
  const result = model.fit(informativeClasses, equalLengths, [1 / 3, 1 / 3, 1 / 3]);
  assert.equal(result.converged, true);
  assert.equal(result.iterations, result.history.length - 1);
  for (let i = 1; i < result.history.length; i += 1) {
    assert.ok(result.history[i].logLikelihood >= result.history[i - 1].logLikelihood - 1e-10);
    close(result.history[i].counts.reduce((a, b) => a + b, 0), 200);
    close(result.history[i].alpha.reduce((a, b) => a + b, 0), 1);
  }
  close(result.alpha[0], 100 / 110, 1e-8);
  close(result.alpha[1], 10 / 110, 1e-8);
  assert.equal(result.alpha[2], 0);
  const limited = model.fit(informativeClasses, equalLengths, [0.5, 0.5, 0], { maxIterations: 1 });
  assert.equal(limited.converged, false);
  assert.equal(limited.iterations, 1);
});

test('unidentifiable equal-length classes preserve starting ratios', () => {
  const classes = [{ members: [0, 1, 2], count: 100 }];
  const left = model.fit(classes, [9, 9, 9], [0.6, 0.3, 0.1]);
  const right = model.fit(classes, [9, 9, 9], [0.1, 0.3, 0.6]);
  vectorClose(left.alpha, [0.6, 0.3, 0.1]);
  vectorClose(right.alpha, [0.1, 0.3, 0.6]);
  close(left.logLikelihood, right.logLikelihood);
});

test('zero-count classes remain valid after an unsupported transcript reaches zero', () => {
  const classes = informativeClasses.concat([
    { members: [2], count: 0 }, { members: [0, 2], count: 0 },
    { members: [1, 2], count: 0 }, { members: [0, 1, 2], count: 0 }
  ]);
  const first = model.emStep(classes, [1 / 3, 1 / 3, 1 / 3], [100, 100, 100]);
  assert.equal(first.alpha[2], 0);
  const nextAllocation = model.expectation(classes, first.alpha, [100, 100, 100]);
  vectorClose(nextAllocation.allocations[3].weights, [0, 0, 0]);
  vectorClose(nextAllocation.allocations[3].expected, [0, 0, 0]);
  const fit = model.fit(classes, [100, 100, 100], first.alpha);
  assert.equal(fit.converged, true);
  close(fit.alpha[0], 100 / 110, 1e-8);
  assert.equal(fit.alpha[2], 0);
  const sampled = model.bootstrapCounts(classes, model.seededRandom(2026));
  const bootstrapFit = model.fit(sampled, [100, 100, 100], [1 / 3, 1 / 3, 1 / 3]);
  assert.equal(bootstrapFit.converged, true);
  assert.equal(bootstrapFit.alpha[2], 0);
});

test('TPM corrects counts for length and sums to one million', () => {
  const result = model.tpm([100, 100, 0], [1000, 2000, 500]);
  vectorClose(result, [2e6 / 3, 1e6 / 3, 0], 1e-8);
  close(result.reduce((a, b) => a + b, 0), 1e6, 1e-8);
  assert.throws(() => model.tpm([0, 0, 0], equalLengths), /undefined/);
});

test('bootstrap is deterministic by seed and conserves the assigned sample size', () => {
  const first = model.bootstrapCounts(informativeClasses, model.seededRandom(2026));
  const second = model.bootstrapCounts(informativeClasses, model.seededRandom(2026));
  assert.deepEqual(first, second);
  assert.equal(first.reduce((sum, group) => sum + group.count, 0), 200);
  assert.deepEqual(first.map(group => group.members), informativeClasses.map(group => group.members));
  assert.notDeepEqual(first, informativeClasses);
  assert.deepEqual(model.bootstrapCounts([], model.seededRandom(1)), []);
  assert.deepEqual(model.bootstrapCounts([{ members: [0], count: 0 }, { members: [1], count: 3 }], () => 0), [
    { members: [0], count: 0 }, { members: [1], count: 3 }
  ]);
  assert.throws(() => model.bootstrapCounts(informativeClasses, () => 1), /\[0, 1\)/);
});

test('fitting rejects empty evidence and invalid parameters rather than inventing estimates', () => {
  assert.throws(() => model.fit([], equalLengths, [1 / 3, 1 / 3, 1 / 3]), /at least one assigned read/);
  assert.throws(() => model.fit([{ members: [0], count: 0 }], equalLengths, [1, 0, 0]), /at least one assigned read/);
  for (const lengths of [[0, 1, 1], [-1, 1, 1], [Infinity, 1, 1]]) {
    assert.throws(() => model.fit(informativeClasses, lengths, [0.5, 0.5, 0]), /finite positive/);
  }
  assert.throws(() => model.fit(informativeClasses, equalLengths, [0, 0, 0]), /sum to one/);
  assert.throws(() => model.fit(informativeClasses, equalLengths, [1, 0, 0]), /zero model probability/);
  assert.throws(() => model.fit([{ members: [3], count: 1 }], equalLengths, [1, 0, 0]), /valid transcript indices/);
});
