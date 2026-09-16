(function () {
  'use strict';
  const M = window.KallistoModel;
  if (!M || !document.getElementById('sequence-lab')) return;
  const $ = id => document.getElementById(id);
  const colors = ['#326da6', '#aa5921', '#408170'];
  const members = [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]];
  const presetClasses = {
    worked: [{members: [0], count: 100}, {members: [1], count: 10}, {members: [0, 1], count: 90}],
    slow: [{members: [0], count: 1}, {members: [0, 1], count: 999}],
    ambiguous: [{members: [0, 1], count: 200}]
  };
  const esc = value => String(value).replace(/[&<>"']/g, char => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[char]));
  const fmt = (n, digits = 2) => n.toLocaleString('en-US', {maximumFractionDigits: digits});
  const setName = set => set.length ? '{' + set.map(t => 'T' + (t + 1)).join(', ') + '}' : '∅';
  const tags = set => set.length ? set.map(t => '<span class="k-tag k-t' + t + '">T' + (t + 1) + '</span>').join(' ') : '∅';
  const sum = values => values.reduce((a, b) => a + b, 0);
  const normalize = values => values.map(v => v / sum(values));
  const defaultReads = () => [
    {sequence: 'ACTGACG', count: 100}, {sequence: 'GACCTA', count: 10},
    {sequence: 'ACTGAC', count: 90}, {sequence: 'GGTGAC', count: 20},
    {sequence: 'TGACGTA', count: 40}
  ];
  const exampleDescriptions = {
    TGACGTA: 'Shared read: the complete sequence occurs in T1 and T3, so the read has more than one possible origin.',
    ACTGACG: 'Unique read: its individual k-mers are shared by different transcripts, but combining them leaves only T1 compatible.',
    ACTGTCGT: 'One substitution: one base differs from the T1 segment ACTGACGT. K-mers crossing that base may be absent, while unaffected k-mers can still provide evidence.',
    AAAAAAA: 'No matches: none of this read’s k-mers occur in the reference index, so the read is unassigned.'
  };
  let reads = defaultReads(), k = 3, selected = 0, index, alignment, aggregate;
  let classes, lengths = [100, 100, 100], initial = [1, 1, 1], alpha, history, pending, iterations;
  let emMessage = '', currentValid = true, animationRun = 0, animating = false, activePhase = '';

  function highlighted(sequence, positions, width) {
    return Array.from(sequence, (base, i) => positions.some(p => i >= p && i < p + width) ? '<mark>' + esc(base) + '</mark>' : esc(base)).join('');
  }

  function updateSequence(resetStep) {
    k = Number($('k-size').value);
    $('k-size-value').textContent = k;
    $('k-class-size').value = k;
    $('k-class-size-value').textContent = k;
    index = M.buildIndex(M.TRANSCRIPTS, k);
    const sequence = $('k-read').value.toUpperCase().replace(/\s/g, '');
    alignment = M.pseudoalign(sequence, k, index);
    document.querySelectorAll('.k-example-reads [data-read]').forEach(button => {
      button.setAttribute('aria-pressed', String(button.dataset.read === sequence));
    });
    $('k-example-description').textContent = exampleDescriptions[sequence] || 'Custom read: follow its k-mers below to see which reference transcripts remain compatible.';
    if (resetStep) selected = 0;
    selected = Math.max(0, Math.min(selected, alignment.steps.length - 1));
    const step = alignment.steps[selected];
    $('k-read').setAttribute('aria-invalid', alignment.status === 'invalid' ? 'true' : 'false');
    $('k-read-window').innerHTML = highlighted(sequence, step ? [step.position] : [], k) || 'Enter a read above.';
    $('k-kmers').innerHTML = alignment.steps.map((s, i) => '<button type="button" data-step="' + i + '" aria-label="k-mer ' + (i + 1) + ': ' + esc(s.kmer) + (s.matched ? '' : ', absent from index') + '" aria-pressed="' + (i === selected) + '" class="' + (s.matched ? '' : 'k-miss') + '">' + esc(s.kmer) + '</button>').join('');
    $('k-prev').disabled = !step || selected === 0;
    $('k-next').disabled = !step || selected === alignment.steps.length - 1;
    $('k-step-label').textContent = step ? 'Window ' + (selected + 1) + ' of ' + alignment.totalKmers : 'No k-mer windows';
    $('k-transcripts').innerHTML = M.TRANSCRIPTS.map((t, i) => {
      const positions = step ? M.kmers(t.sequence, k).filter(s => s.kmer === step.kmer).map(s => s.position) : [];
      return '<div class="k-transcript"><span class="k-tag k-t' + i + '">' + t.id + '</span><code>' + highlighted(t.sequence, positions, k) + '</code><span class="k-help">' + (step ? (positions.length ? 'contains ' + esc(step.kmer) : 'no ' + esc(step.kmer)) : '') + '</span></div>';
    }).join('');
    if (step) {
      const seenHit = alignment.steps.slice(0, selected + 1).some(s => s.matched);
      const previousHit = alignment.steps.slice(0, selected).some(s => s.matched);
      const before = previousHit ? setName(alignment.steps[selected - 1].remaining) : 'all transcripts';
      $('k-intersection').innerHTML = step.matched
        ? '<strong>' + esc(step.kmer) + '</strong> → ' + tags(step.compatible) + '<br>Intersection so far: ' + esc(before) + ' ∩ ' + esc(setName(step.compatible)) + ' = <strong>' + esc(setName(step.remaining)) + '</strong>'
        : '<strong>' + esc(step.kmer) + '</strong> is absent from the index. Skip this k-mer.<br>' + (seenHit ? 'Intersection stays ' + esc(setName(step.remaining)) + '.' : 'No matching evidence yet; no transcript can be assigned.');
    } else {
      $('k-intersection').textContent = alignment.status === 'too-short' ? 'This read is shorter than k. Lower k or enter a longer read.' : 'Enter a read using only A, C, G, and T (up to 60 bases).';
    }
    const descriptions = {
      matched: 'Final read compatibility: ' + setName(alignment.compatible) + '.',
      'no-hits': 'Unassigned: no read k-mer occurs in the index.',
      conflict: 'Unassigned: matching k-mers contradict one another; their intersection is empty.',
      'too-short': 'Unassigned: read length is less than k.',
      invalid: 'Unassigned: the read contains invalid bases or is empty.'
    };
    $('k-read-result').textContent = descriptions[alignment.status] + ' ' + alignment.matchedCount + ' / ' + alignment.totalKmers + ' windows match the index. A matching window is evidence, not a separate read count.';
    $('k-index-summary').textContent = index.size + ' distinct ' + k + '-mers; repeated occurrences share one entry. Highlighted row = selected read k-mer.';
    $('k-index').innerHTML = Array.from(index).sort((a, b) => a[0].localeCompare(b[0])).map(([mer, set]) => '<tr class="' + (step && mer === step.kmer ? 'k-index-active' : '') + '"><td><code>' + mer + '</code></td><td>' + tags(set) + '</td></tr>').join('');
    updateSample();
  }

  function updateSample() {
    aggregate = M.aggregate(reads, k, index);
    $('k-read-matrix').innerHTML = reads.map((read, i) => '<tr><td><code>' + esc(read.sequence) + '</code></td><td><input type="number" data-count="' + i + '" aria-label="Copies of ' + esc(read.sequence) + '" min="0" max="10000" step="1" value="' + read.count + '"></td>' + [0, 1, 2].map(t => {
      const compatible = aggregate.results[i].compatible.includes(t);
      return '<td class="k-compat-' + (compatible ? 'yes k-compat-t' + t : 'no') + '" data-compatible="' + compatible + '" aria-label="' + (compatible ? 'Compatible' : 'Not compatible') + ' with T' + (t + 1) + '">' + (compatible ? '1' : '0') + '</td>';
    }).join('') + '</tr>').join('');
    const total = aggregate.assigned + aggregate.unassigned;
    $('k-sample-totals').textContent = fmt(total, 0) + ' reads → ' + fmt(aggregate.assigned, 0) + ' assigned + ' + fmt(aggregate.unassigned, 0) + ' unassigned → ' + aggregate.classes.filter(e => e.count > 0).length + ' nonempty equivalence classes (k = ' + k + ').';
    $('k-classes').innerHTML = aggregate.classes.filter(e => e.count > 0).map(e => '<div class="k-class">' + tags(e.members) + '<br><strong>' + fmt(e.count, 0) + '</strong> fragments</div>').join('') || '<p class="k-help">No assigned fragments yet. Increase a copy count or restore the sample.</p>';
    $('k-use-sample').disabled = aggregate.assigned === 0;
  }

  function fullClasses(source) {
    return members.map(set => ({members: set.slice(), count: source.find(e => setName(e.members) === setName(set))?.count || 0}));
  }

  function parameterInputs() {
    $('k-em-counts').innerHTML = classes.map((e, i) => '<label for="k-count-' + i + '">' + esc(setName(e.members)) + '<input id="k-count-' + i + '" data-ec="' + i + '" type="number" min="0" max="200000" step="1" value="' + e.count + '"></label>').join('');
    $('k-em-parameters').innerHTML = [0, 1, 2].map(t => '<tr><th scope="row">T' + (t + 1) + '</th><td><input data-length="' + t + '" type="number" min="1" max="10000" step="1" value="' + lengths[t] + '" aria-label="T' + (t + 1) + ' effective length"></td><td><input data-initial="' + t + '" type="number" min="0.01" max="100" step="0.01" value="' + initial[t] + '" aria-label="T' + (t + 1) + ' starting weight"></td></tr>').join('');
  }

  function resetEM(message = '') {
    animationRun++;
    animating = false;
    activePhase = '';
    pending = null; iterations = 0; alpha = normalize(initial);
    currentValid = sum(classes.map(e => e.count)) > 0;
    history = [{alpha: alpha.slice(), logLikelihood: currentValid ? M.likelihood(classes, alpha, lengths) : 0}];
    emMessage = message || 'Initialized. First, use the E-step to split each class using the current α / ℓ.';
    renderEM();
  }

  function loadPreset(name) {
    lengths = [100, 100, 100]; initial = [1, 1, 1];
    if (name === 'custom') return;
    classes = fullClasses(presetClasses[name]);
    parameterInputs();
    resetEM(name === 'ambiguous' ? 'Only {T1, T2} has evidence. Compare equal initialization with “Start favoring T1”: the data cannot distinguish those splits.' : '');
  }

  function renderEM() {
    const n = sum(classes.map(e => e.count));
    const counts = alpha.map(a => a * n);
    const tpm = currentValid ? M.tpm(counts, lengths) : [0, 0, 0];
    $('k-em-status').setAttribute('aria-live', animating ? 'off' : 'polite');
    $('k-em-status').textContent = currentValid ? 'Iteration ' + iterations + '. ' + emMessage : 'No observations. Add a positive class count before fitting. There is no abundance estimate from an empty sample.';
    $('k-em-class-summary').textContent = 'Current input: ' + classes.filter(e => e.count > 0).map(e => setName(e.members) + ' × ' + e.count).join('; ') + '. N = ' + n + '.';
    if (!$('k-em-bars').firstElementChild) $('k-em-bars').innerHTML = [0, 1, 2].map(t => '<div class="k-bar-row"><strong>T' + (t + 1) + '</strong><div class="k-bar-track"><div class="k-bar-fill" data-em-bar="' + t + '" style="background:' + colors[t] + '"></div></div><span class="k-number" data-em-value="' + t + '"></span></div>').join('');
    [0, 1, 2].forEach(t => {
      $('k-em-bars').querySelector('[data-em-bar="' + t + '"]').style.width = (currentValid ? alpha[t] * 100 : 0) + '%';
      $('k-em-bars').querySelector('[data-em-value="' + t + '"]').textContent = currentValid ? fmt(alpha[t] * 100, 2) + '%' : '—';
    });
    $('k-em-estimates').innerHTML = [0, 1, 2].map(t => '<tr><th scope="row">T' + (t + 1) + '</th><td class="k-number">' + (currentValid ? alpha[t].toFixed(5) : '—') + '</td><td class="k-number">' + (currentValid ? fmt(counts[t], 3) : '—') + '</td><td class="k-number">' + (currentValid ? fmt(tpm[t], 1) : '—') + '</td></tr>').join('');
    $('k-e-step').hidden = !pending;
    if (pending) {
      $('k-e-allocations').innerHTML = pending.allocations.filter(e => e.count > 0).map(e => '<tr><th scope="row">' + setName(e.members) + ' (' + e.count + ')</th>' + e.expected.map((value, t) => '<td class="k-number">' + fmt(value, 3) + '<br><span class="k-help">w = ' + fmt(e.weights[t], 4) + '</span></td>').join('') + '</tr>').join('');
      $('k-e-totals').innerHTML = '<tr><th scope="row">Expected counts</th>' + pending.counts.map(value => '<td><strong>' + fmt(value, 3) + '</strong></td>').join('') + '</tr>';
    }
    $('k-em-step').textContent = pending ? 'M-step: update α' : 'E-step: split counts';
    $('k-em-animate').textContent = animating ? 'Stop animation' : 'Animate E/M from start';
    $('k-em-animate').setAttribute('aria-pressed', String(animating));
    $('k-em-animate').disabled = !currentValid;
    $('k-em-phase-e').dataset.active = String(activePhase === 'e');
    $('k-em-phase-m').dataset.active = String(activePhase === 'm');
    ['k-em-step', 'k-em-run'].forEach(id => $(id).disabled = !currentValid || animating);
    plotHistory();
    const ll = history[history.length - 1].logLikelihood;
    $('k-em-likelihood').textContent = currentValid ? 'Log likelihood (natural log; constant omitted): ' + ll.toFixed(6) + '. Change from initialization: +' + (ll - history[0].logLikelihood).toFixed(6) + '. Compare likelihoods only for the same counts and lengths.' : '';
  }

  function plotHistory() {
    const last = history.length - 1, width = 528, height = 125, left = 44, top = 12;
    let svg = '<title>Fragment share α across EM iterations</title>';
    [0, .5, 1].forEach(v => { const y = top + (1 - v) * height; svg += '<line x1="44" x2="572" y1="' + y + '" y2="' + y + '" stroke="#dce2e6"/><text x="4" y="' + (y + 4) + '">' + v * 100 + '%</text>'; });
    const stride = Math.max(1, Math.ceil(history.length / 250));
    [0, 1, 2].forEach(t => {
      const samples = history.map((h, i) => ({h, i})).filter(({i}) => i % stride === 0 || i === last);
      const points = samples.map(({h, i}) => (left + i / Math.max(1, last) * width).toFixed(2) + ',' + (top + (1 - h.alpha[t]) * height).toFixed(2)).join(' ');
      if (currentValid) svg += '<polyline points="' + points + '" fill="none" stroke="' + colors[t] + '" stroke-width="2"' + (t ? ' stroke-dasharray="' + (t === 1 ? '7 3' : '2 3') + '"' : '') + '/><circle cx="' + (left + (last ? width : 0)) + '" cy="' + (top + (1 - alpha[t]) * height) + '" r="3" fill="' + colors[t] + '"/>';
      svg += '<line x1="' + (160 + t * 90) + '" x2="' + (180 + t * 90) + '" y1="181" y2="181" stroke="' + colors[t] + '" stroke-width="3"/><text x="' + (186 + t * 90) + '" y="185">T' + (t + 1) + '</text>';
    });
    svg += '<text x="44" y="157">0</text><text x="' + (last ? 560 : 80) + '" y="157">' + (last || '') + '</text><text x="240" y="157">EM iteration</text>';
    $('k-em-chart').innerHTML = svg;
    $('k-em-chart-caption').textContent = currentValid ? 'Watch the fragment shares settle. Overlapping curves may hide one another; the values above give the exact current split.' : 'No trajectory to plot for an empty sample.';
  }

  function oneStep() {
    if (!currentValid) return;
    if (!pending) {
      pending = M.expectation(classes, alpha, lengths);
      activePhase = 'e';
      emMessage = 'E-step complete. The expected assignments below use the current estimate. Now apply the M-step.';
    } else {
      const next = M.emStep(classes, alpha, lengths);
      alpha = next.alpha; history.push(next); iterations++; pending = null;
      activePhase = 'm';
      emMessage = 'M-step complete. Maximum change in α = ' + next.delta.toExponential(2) + (next.delta < 1e-8 ? '; the stopping tolerance is satisfied.' : '. Take another E-step or run the remaining iterations.');
    }
    renderEM();
  }

  function runEM() {
    if (!currentValid) return;
    const result = M.fit(classes, lengths, alpha, {maxIterations: 2000, tolerance: 1e-8});
    alpha = result.alpha; iterations += result.iterations; pending = null;
    history.push(...result.history.slice(1));
    activePhase = 'm';
    emMessage = result.converged ? 'Stopped: maximum change in α is below 10⁻⁸. Stability alone does not establish identifiability.' : 'Reached the 2,000-update limit for this run; not converged to the selected tolerance. You can continue running.';
    renderEM();
  }

  function animationCheckpoints(total) {
    if (total <= 12) return Array.from({length: total}, (_, i) => i + 1);
    const points = [1, 2, 3, 4, 5, 6];
    for (let i = 1; i <= 6; i++) points.push(Math.round(6 * Math.pow(total / 6, i / 6)));
    return Array.from(new Set(points)).filter(point => point <= total).sort((a, b) => a - b);
  }

  async function animateEM() {
    if (!currentValid) return;
    if (animating) {
      animationRun++;
      animating = false;
      activePhase = '';
      emMessage = 'Animation stopped at iteration ' + iterations + '. Continue manually or restart from the beginning.';
      renderEM();
      return;
    }

    resetEM('Animation starting from the selected initial weights.');
    const result = M.fit(classes, lengths, alpha, {maxIterations: 2000, tolerance: 1e-8});
    const checkpoints = animationCheckpoints(result.iterations);
    const token = animationRun;
    animating = true;
    renderEM();
    await new Promise(resolve => setTimeout(resolve, 250));

    for (let i = 0; i < checkpoints.length; i++) {
      if (token !== animationRun) return;
      const checkpoint = checkpoints[i];
      const skipped = i > 0 && checkpoint > checkpoints[i - 1] + 1;
      alpha = result.history[checkpoint - 1].alpha.slice();
      history = result.history.slice(0, checkpoint);
      iterations = checkpoint - 1;
      pending = M.expectation(classes, alpha, lengths);
      activePhase = 'e';
      emMessage = (skipped ? 'Fast-forwarded to iteration ' + iterations + '. ' : '') + 'E-step ' + checkpoint + ': divide every class using the current α / ℓ weights.';
      renderEM();
      await new Promise(resolve => setTimeout(resolve, 350));

      if (token !== animationRun) return;
      const next = result.history[checkpoint];
      alpha = next.alpha.slice();
      history = result.history.slice(0, checkpoint + 1);
      iterations = checkpoint;
      pending = null;
      activePhase = 'm';
      emMessage = 'M-step ' + checkpoint + ': update α from the expected counts. Maximum change = ' + next.delta.toExponential(2) + '.';
      renderEM();
      await new Promise(resolve => setTimeout(resolve, 250));
    }

    if (token !== animationRun) return;
    animating = false;
    activePhase = '';
    emMessage = result.converged
      ? 'Animation complete: converged after ' + result.iterations + ' updates. Maximum change is below 10⁻⁸.'
      : 'Animation complete: reached the 2,000-update limit without meeting the stopping tolerance.';
    renderEM();
  }

  function validNumber(input, min, max, integer) {
    const value = Number(input.value);
    if (!input.value.trim() || !Number.isFinite(value) || value < min || value > max || (integer && !Number.isInteger(value))) return null;
    return value;
  }

  $('k-read').addEventListener('input', () => updateSequence(true));
  $('k-size').addEventListener('input', () => updateSequence(true));
  $('k-class-size').addEventListener('input', event => {
    $('k-size').value = event.target.value;
    updateSequence(true);
    $('k-sample-message').textContent = 'Reprocessed the sample with k = ' + k + '.';
  });
  $('sequence-lab').addEventListener('click', event => {
    const button = event.target.closest('button');
    if (!button) return;
    if (button.dataset.read) { $('k-read').value = button.dataset.read; updateSequence(true); }
    if (button.dataset.step !== undefined) { selected = Number(button.dataset.step); updateSequence(false); }
  });
  $('k-prev').addEventListener('click', () => { selected--; updateSequence(false); });
  $('k-next').addEventListener('click', () => { selected++; updateSequence(false); });
  $('k-reset-reads').addEventListener('click', () => {
    reads = defaultReads();
    $('k-size').value = '3';
    updateSequence(true);
    $('k-sample-message').textContent = 'Restored the default counts and k = 3.';
  });
  $('k-read-matrix').addEventListener('change', event => {
    const input = event.target;
    if (input.dataset.count === undefined) return;
    const value = validNumber(input, 0, 10000, true);
    if (value === null) { input.value = reads[input.dataset.count].count; $('k-sample-message').textContent = 'Use a whole-number count from 0 to 10,000. Previous count restored.'; return; }
    reads[input.dataset.count].count = value;
    $('k-sample-message').textContent = 'Count updated.'; updateSample();
  });
  $('k-use-sample').addEventListener('click', () => {
    classes = fullClasses(aggregate.classes); lengths = [100, 100, 100]; initial = [1, 1, 1];
    $('k-em-preset').value = 'custom'; parameterInputs(); resetEM('Loaded ' + aggregate.assigned + ' assigned fragments from experiment 2 (k = ' + k + ').');
    $('em-lab').scrollIntoView({behavior: 'auto', block: 'start'}); $('k-em-step').focus({preventScroll: true});
  });
  $('k-em-preset').addEventListener('change', event => { if (event.target.value === 'custom') { $('em-lab').querySelector('details').open = true; } else loadPreset(event.target.value); });
  $('em-lab').addEventListener('change', event => {
    const input = event.target;
    let key, target, min, max, integer;
    if (input.dataset.ec !== undefined) { key = Number(input.dataset.ec); target = 'count'; min = 0; max = 200000; integer = true; }
    else if (input.dataset.length !== undefined) { key = Number(input.dataset.length); target = 'length'; min = 1; max = 10000; integer = true; }
    else if (input.dataset.initial !== undefined) { key = Number(input.dataset.initial); target = 'initial'; min = .01; max = 100; integer = false; }
    else return;
    const value = validNumber(input, min, max, integer);
    if (value === null) {
      input.value = target === 'count' ? classes[key].count : target === 'length' ? lengths[key] : initial[key];
      $('k-em-status').textContent = 'Use ' + (integer ? 'a whole number' : 'a number') + ' from ' + min + ' to ' + max + '. Previous value restored.';
      return;
    }
    if (target === 'count') classes[key].count = value;
    if (target === 'length') lengths[key] = value;
    if (target === 'initial') initial[key] = value;
    $('k-em-preset').value = 'custom'; resetEM('Parameters changed; EM restarted.');
  });
  $('k-em-step').addEventListener('click', oneStep);
  $('k-em-animate').addEventListener('click', animateEM);
  $('k-em-run').addEventListener('click', runEM);
  $('k-em-reset').addEventListener('click', () => resetEM());
  $('k-em-favor').addEventListener('click', () => { initial = [8, 1, 1]; parameterInputs(); resetEM('Starting weights are now 8:1:1. Compare the result with equal starting weights (set them back to 1:1:1 under Edit).'); });
  updateSequence(true);
  loadPreset('worked');
}());
