/* A small, deliberately forward-strand-only teaching model, not kallisto itself. */
(function (root, factory) {
  "use strict";
  var model = factory();
  if (typeof module === "object" && module.exports) module.exports = model;
  if (root) root.KallistoModel = model;
})(typeof window !== "undefined" ? window : null, function () {
  "use strict";

  var TRANSCRIPTS = [
    { id: "T1", sequence: "ACTGACGTA" },
    { id: "T2", sequence: "ACTGACCTA" },
    { id: "T3", sequence: "GGTGACGTA" }
  ];

  function normalize(sequence) {
    return typeof sequence === "string" ? sequence.trim().toUpperCase() : "";
  }

  function validK(k) {
    return Number.isInteger(k) && k > 0;
  }

  function kmers(sequence, k) {
    if (!validK(k)) throw new Error("k must be a positive integer.");
    var normalized = normalize(sequence);
    if (!/^[ACGT]*$/.test(normalized)) throw new Error("Sequences must contain only A, C, G, and T.");
    var result = [];
    for (var i = 0; i <= normalized.length - k; i += 1) {
      result.push({ kmer: normalized.slice(i, i + k), position: i });
    }
    return result;
  }

  function buildIndex(transcripts, k) {
    if (!Array.isArray(transcripts)) throw new Error("Provide an array of transcripts.");
    if (!validK(k)) throw new Error("k must be a positive integer.");
    var index = new Map();
    transcripts.forEach(function (transcript, transcriptIndex) {
      kmers(transcript.sequence, k).forEach(function (item) {
        var members = index.get(item.kmer);
        if (!members) {
          members = [];
          index.set(item.kmer, members);
        }
        // Repeated occurrences within one transcript contribute one membership.
        if (members.indexOf(transcriptIndex) === -1) members.push(transcriptIndex);
      });
    });
    return index;
  }

  function pseudoalign(read, k, index) {
    var result = {
      steps: [], compatible: [], status: "no-hits", matchedCount: 0, totalKmers: 0
    };
    var sequence = normalize(read);
    if (!validK(k) || !/^[ACGT]+$/.test(sequence)) {
      result.status = "invalid";
      return result;
    }
    if (sequence.length < k) {
      result.status = "too-short";
      return result;
    }
    if (!(index instanceof Map)) throw new Error("Provide a k-mer index from buildIndex.");
    var remaining = null;
    var words = kmers(sequence, k);
    result.totalKmers = words.length;
    words.forEach(function (word) {
      var compatible = (index.get(word.kmer) || []).slice();
      var matched = compatible.length > 0;
      if (matched) {
        result.matchedCount += 1;
        remaining = remaining === null ? compatible.slice() : remaining.filter(function (member) {
          return compatible.indexOf(member) !== -1;
        });
      }
      result.steps.push({
        kmer: word.kmer,
        position: word.position,
        compatible: compatible,
        remaining: remaining === null ? [] : remaining.slice(),
        matched: matched
      });
    });
    result.compatible = remaining === null ? [] : remaining;
    result.status = !result.matchedCount ? "no-hits" : result.compatible.length ? "matched" : "conflict";
    return result;
  }

  function aggregate(reads, k, index) {
    if (!Array.isArray(reads)) throw new Error("Provide an array of reads and counts.");
    var grouped = new Map();
    var assigned = 0;
    var unassigned = 0;
    var results = reads.map(function (read) {
      if (!Number.isInteger(read.count) || read.count < 0) {
        throw new Error("Read counts must be nonnegative integers.");
      }
      var result = pseudoalign(read.sequence, k, index);
      if (result.status === "matched") {
        assigned += read.count;
        var key = result.compatible.join(",");
        if (!grouped.has(key)) grouped.set(key, { members: result.compatible.slice(), count: 0 });
        grouped.get(key).count += read.count;
      } else {
        unassigned += read.count;
      }
      return result;
    });
    return {
      classes: Array.from(grouped.values()).filter(function (group) { return group.count > 0; }),
      assigned: assigned,
      unassigned: unassigned,
      results: results
    };
  }

  function validateLengths(lengths) {
    if (!Array.isArray(lengths) || !lengths.length || lengths.some(function (length) {
      return !Number.isFinite(length) || length <= 0;
    })) throw new Error("Effective lengths must be finite positive numbers.");
  }

  function validateClasses(classes, dimensions) {
    if (!Array.isArray(classes)) throw new Error("Provide an array of compatibility classes.");
    classes.forEach(function (group) {
      if (!group || !Number.isFinite(group.count) || group.count < 0) {
        throw new Error("Class counts must be finite nonnegative numbers.");
      }
      if (!Array.isArray(group.members) || !group.members.length || group.members.some(function (member, i) {
        return !Number.isInteger(member) || member < 0 || member >= dimensions || group.members.indexOf(member) !== i;
      })) throw new Error("Each compatibility class must contain distinct valid transcript indices.");
    });
  }

  function validateAlpha(alpha, dimensions) {
    if (!Array.isArray(alpha) || alpha.length !== dimensions || alpha.some(function (value) {
      return !Number.isFinite(value) || value < 0;
    })) throw new Error("Abundances must be finite nonnegative numbers, one per transcript.");
    var total = alpha.reduce(function (sum, value) { return sum + value; }, 0);
    if (Math.abs(total - 1) > 1e-8) throw new Error("Fragment abundances must sum to one.");
  }

  function validateModel(classes, alpha, lengths) {
    validateLengths(lengths);
    validateClasses(classes, lengths.length);
    validateAlpha(alpha, lengths.length);
  }

  function observationCount(classes) {
    return classes.reduce(function (sum, group) { return sum + group.count; }, 0);
  }

  // Log likelihood, omitting factors constant in alpha. alpha is fragment share;
  // it is not the length-corrected molecule share reported as TPM.
  function likelihood(classes, alpha, lengths) {
    validateModel(classes, alpha, lengths);
    return classes.reduce(function (sum, group) {
      if (group.count === 0) return sum;
      var probability = group.members.reduce(function (mass, member) {
        return mass + alpha[member] / lengths[member];
      }, 0);
      return sum + group.count * Math.log(probability);
    }, 0);
  }

  function expectation(classes, alpha, lengths) {
    validateModel(classes, alpha, lengths);
    var counts = alpha.map(function () { return 0; });
    var allocations = classes.map(function (group) {
      var weights = alpha.map(function () { return 0; });
      var expected = alpha.map(function () { return 0; });
      var denominator = group.members.reduce(function (sum, member) {
        return sum + alpha[member] / lengths[member];
      }, 0);
      if (denominator === 0 && group.count > 0) {
        throw new Error("A counted class has zero model probability. Start with positive abundance for its transcripts.");
      }
      if (denominator > 0) group.members.forEach(function (member) {
        weights[member] = (alpha[member] / lengths[member]) / denominator;
        expected[member] = group.count * weights[member];
        counts[member] += expected[member];
      });
      return { members: group.members.slice(), count: group.count, weights: weights, expected: expected };
    });
    return { allocations: allocations, counts: counts };
  }

  function emStep(classes, alpha, lengths) {
    var allocation = expectation(classes, alpha, lengths);
    var total = observationCount(classes);
    if (!(total > 0)) throw new Error("Add at least one assigned read before fitting abundances.");
    var next = allocation.counts.map(function (count) { return count / total; });
    // Teaching stop rule: the largest absolute transcript-share change.
    var delta = next.reduce(function (largest, value, i) { return Math.max(largest, Math.abs(value - alpha[i])); }, 0);
    return {
      alpha: next,
      counts: allocation.counts,
      logLikelihood: likelihood(classes, next, lengths),
      delta: delta
    };
  }

  function fit(classes, lengths, initial, options) {
    options = options || {};
    var maxIterations = options.maxIterations === undefined ? 1000 : options.maxIterations;
    var tolerance = options.tolerance === undefined ? 1e-8 : options.tolerance;
    if (!Number.isInteger(maxIterations) || maxIterations < 1) throw new Error("maxIterations must be a positive integer.");
    if (!Number.isFinite(tolerance) || tolerance < 0) throw new Error("tolerance must be finite and nonnegative.");
    validateModel(classes, initial, lengths);
    var total = observationCount(classes);
    if (!(total > 0)) throw new Error("Add at least one assigned read before fitting abundances.");
    var current = {
      alpha: initial.slice(),
      counts: initial.map(function (value) { return value * total; }),
      logLikelihood: likelihood(classes, initial, lengths),
      delta: 0,
      iteration: 0
    };
    var history = [current];
    var converged = false;
    for (var i = 1; i <= maxIterations; i += 1) {
      current = emStep(classes, current.alpha, lengths);
      current.iteration = i;
      history.push(current);
      if (current.delta < tolerance) {
        converged = true;
        break;
      }
    }
    return {
      alpha: current.alpha.slice(),
      counts: current.counts.slice(),
      iterations: history.length - 1,
      converged: converged,
      logLikelihood: current.logLikelihood,
      history: history
    };
  }

  function tpm(counts, lengths) {
    validateLengths(lengths);
    if (!Array.isArray(counts) || counts.length !== lengths.length || counts.some(function (value) {
      return !Number.isFinite(value) || value < 0;
    })) throw new Error("Counts must be finite nonnegative numbers, one per transcript.");
    var rates = counts.map(function (count, i) { return count / lengths[i]; });
    var total = rates.reduce(function (sum, value) { return sum + value; }, 0);
    if (!(total > 0)) throw new Error("TPM is undefined when there are no assigned counts.");
    return rates.map(function (rate) { return 1e6 * rate / total; });
  }

  return {
    TRANSCRIPTS: TRANSCRIPTS,
    kmers: kmers,
    buildIndex: buildIndex,
    pseudoalign: pseudoalign,
    aggregate: aggregate,
    likelihood: likelihood,
    expectation: expectation,
    emStep: emStep,
    fit: fit,
    tpm: tpm
  };
});
