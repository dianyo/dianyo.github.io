---
layout: post
title: "Understanding kallisto: from k-mers to abundance, with experiments"
permalink: /understanding-kallisto/
excerpt: "Build a k-mer index, pseudoalign reads, compress compatibility classes, and step through EM. Then reconsider the algorithm on a GPU."
---

How can we estimate the abundance of RNA transcripts without aligning every base of every read? Kallisto makes this question concrete. In this article, we will build a small version of its reasoning: turn a read into a set of possible transcripts, collect those sets, and use them to estimate abundance. Each experiment runs locally in your browser.

<link rel="stylesheet" href="{{ site.baseurl }}/assets/kallisto/article.css">

This started with [IRIC’s explanation of kallisto](https://bioinfo.iric.ca/understanding-how-kallisto-works/). I wanted to go further into three questions: what information does the compression lose, what does EM actually optimize, and would modern GPUs change the algorithm we choose? The experiments below are my own small teaching implementation, not kallisto compiled for the browser.

<div class="k-roadmap" aria-label="Article contents">
  <strong>Read it in order, or jump into an experiment</strong>
  <ol>
    <li><a href="#the-problem">What are we trying to count?</a></li>
    <li><a href="#sequence-lab">Play with k-mers and pseudoalignment</a></li>
    <li><a href="#class-lab">Build compatibility classes</a></li>
    <li><a href="#em-lab">Step through EM</a></li>
    <li><a href="#bootstrap-lab">Resample the data</a></li>
    <li><a href="#gpu">What changes on a GPU?</a></li>
  </ol>
</div>

<noscript><p class="k-note">JavaScript is disabled. The article and worked calculations remain readable, but the experiments require JavaScript.</p></noscript>

<h2 id="the-problem">1. The problem: one fragment, several possible origins</h2>

A gene can produce several RNA transcripts through alternative splicing. Those transcripts, or isoforms, share some sequence and differ elsewhere. An RNA-seq experiment samples fragments from the RNA population; a short sequenced read may cover only a shared region. A paired-end experiment reads both ends of a fragment. The pair is one observation of origin, not two independent molecules.

Suppose we have three tiny reference transcripts. These nine-base strings are invented so we can inspect every operation; real transcripts and reads are much longer.

```text
T1   ACTGACGTA
T2   ACTGACCTA
T3   GGTGACGTA
```

The read `TGACGTA` occurs in both T1 and T3. Even a perfect alignment cannot identify its origin uniquely. Discarding ambiguous reads wastes evidence; counting each candidate as a whole read inflates the total. We need a way to distribute that evidence.

Kallisto separates the work into two stages: pseudoalignment determines candidate transcripts, then a statistical model estimates how much each transcript contributed. That separation is the central idea in [Bray, Pimentel, Melsted, and Pachter’s original paper](https://www.nature.com/articles/nbt.3519).

<div class="k-flow" aria-label="Algorithm stages"><span>Read sequences</span><span>→ candidate sets</span><span>→ class counts</span><span>→ abundance</span></div>

The output is an estimate of expression relative to a supplied reference transcriptome. This procedure does not assemble missing isoforms or discover all the RNA that might exist in a cell.

## 2. K-mers: make sequence searchable

A **k-mer** is a contiguous substring of k bases. A sequence of length L has L − k + 1 overlapping windows when L ≥ k. For `TGACGTA` with k = 3:

```text
TGA
 GAC
  ACG
   CGT
    GTA
```

We can index the reference by associating each distinct k-mer with the transcripts containing it. In our example, `TGA` occurs in all three transcripts, while `ACG` occurs in T1 and T3. Repeated appearances of a k-mer within a transcript do not create additional transcript identities in this toy set.

**Try this:** leave k at 3 and select **Shared read**. Click `ACG`. Watch T2 disappear from the running intersection. Then select **Unique read**: neither the first nor the last part identifies T1 alone, but their combination does.

{% include kallisto/sequence-lab.html %}

The operation is set intersection:

<div class="k-equation">
  C(TGACGTA) = C(TGA) ∩ C(GAC) ∩ C(ACG) ∩ C(CGT) ∩ C(GTA)<br>
  = {T1, T2, T3} ∩ {T1, T2, T3} ∩ {T1, T3} ∩ {T1, T3} ∩ {T1, T3}<br>
  = {T1, T3}
</div>

Intersecting asks which transcripts survive **all the matching evidence**. A union would retain a transcript supported by even one piece, regardless of contradictory pieces. Also notice that the five windows still represent **one read**. We do not turn their overlapping evidence into five independent observations for EM.

### What changing k teaches us

Small k-mers occur more readily by chance. Longer k-mers can distinguish sequences better, but give a short read fewer windows and allow a single substitution to disrupt more windows. Try **One substitution** at k = 3, then increase k. Which unaffected windows remain? Does the read still have any usable evidence?

In this demo, an absent k-mer is skipped. No matching k-mers means **unassigned**, and contradictory matching k-mers can also leave an empty set. Skipping unknown words never turns “no evidence” into “compatible with everything.” An error can create a k-mer that exists elsewhere, so this is not a guarantee of error correction.

We deliberately allow k = 2–7 for exploration and search the forward strand only. Production kallisto handles orientation, paired ends, and additional constraints; its documented index default is k = 31 with an odd-k requirement. Do not treat our slider range as a recommendation for real data. See the [kallisto manual](https://pachterlab.github.io/kallisto/manual).

### Where the graph fits

The full index is more than our lookup table. In the original transcriptome de Bruijn graph, k-mers are nodes, neighboring sequence windows connect them, and transcript membership supplies the colors. A transcript follows a path. Linear stretches with unchanged membership can be compacted into contigs; a contig is not necessarily a whole transcript. Kallisto uses this structure to skip redundant lookups and checks the end of a skip. Our demo examines every window to make the intersection visible. These details are described in the [original manuscript’s indexing and pseudoalignment methods](https://arxiv.org/pdf/1505.02710).

Our table also ignores order beyond individual k-mer matches. With tiny k, a string can pass a set-intersection test without occurring as a contiguous substring. Pseudoalignment provides compatibility evidence; it is not a base-by-base alignment certificate.

## 3. Compatibility classes: count the repeated questions

Three related terms are easy to confuse:

- A **k-mer compatibility set** contains transcripts that contain that k-mer.
- A **read or fragment compatibility set** contains candidates surviving the combined evidence.
- An **equivalence class (EC)** groups observations with the same final candidate set. Its count is the number of observations in that group. These are also called transcript compatibility counts (TCCs).

For example, `{T1, T2}: 90` means ninety fragments have that candidate set. It does not mean ninety fragments have already been assigned to each transcript.

**Try this:** adjust the copies below, then change k in experiment 1. You may move reads between classes or lose assignments. Add an unmatched read and check that the unassigned total increases without adding an EM class. Finally, press **Use these class counts in EM** to carry your sample forward.

{% include kallisto/class-lab.html %}

### Is this compression one-to-one?

No. Consider four labeled observations:

```text
          T1  T2  T3
read A     1   1   0
read B     1   1   0
read C     1   0   0
read D     1   1   0
```

The grouped version is `{T1, T2}: 3` and `{T1}: 1`. Read identities and ordering are gone. Many labeled matrices yield those counts.

But we can ask a narrower question: **does grouping identical rows change the objective we optimize?** Let α<sub class="k-sub">t</sub> be the probability that a sampled fragment comes from transcript t, and ℓ<sub class="k-sub">t</sub> its fixed effective length. The simplified model uses this likelihood:

<div class="k-equation">
  g<sub class="k-sub">e</sub>(α) = ∑<sub class="k-sub">t ∈ e</sub> α<sub class="k-sub">t</sub> / ℓ<sub class="k-sub">t</sub><br>
  L(α) ∝ ∏<sub class="k-sub">f</sub> g<sub class="k-sub">C(f)</sub>(α) = ∏<sub class="k-sub">e</sub> g<sub class="k-sub">e</sub>(α)<sup>c<sub class="k-sub">e</sub></sup><br>
  α<sub class="k-sub">t</sub> ≥ 0, &nbsp; ∑<sub class="k-sub">t</sub> α<sub class="k-sub">t</sub> = 1
</div>

For our four rows, both representations give `g₁₂ × g₁₂ × g₁ × g₁₂ = g₁₂³ × g₁`. The equality is ordinary factorization, with no numerical approximation. EC counts are sufficient for this fixed likelihood. The [original quantification methods](https://arxiv.org/pdf/1505.02710) state the model in this form.

This separates two different losses. Reducing sequence evidence to binary compatibility can discard information useful to a richer model. Grouping identical compatibility rows preserves the displayed objective exactly. If we later use fragment-specific alignment scores, positional biases, or different weights within the same candidate set, we must revisit whether those counts alone are sufficient.

## 4. Why effective length appears

Imagine two transcripts with equal numbers of RNA molecules, but one offers twice as many possible fragment starts. It can contribute more fragments without having more molecules. We therefore distinguish **fragment share α** from **molecular relative abundance**.

For a transcript of length L and a fixed fragment length d ≤ L, a simple effective length is `ℓ = L − d + 1`. Real library models account for a fragment-length distribution. For background on these modeling choices, see Pachter’s [Models for transcript quantification from RNA-Seq](https://arxiv.org/abs/1104.3889).

In our simplified generative picture, we first select transcript t with probability α<sub class="k-sub">t</sub>, then select a possible fragment location. The second choice contributes the inverse-length factor. This is why a compatible transcript contributes `α / ℓ` to the likelihood term.

Here is a useful check: with equal fragment shares but effective lengths 100 and 200, a shared observation has relative weights `(0.5 / 100) : (0.5 / 200) = 2 : 1`. You can try exactly this change in the EM controls. These length controls are independent statistical parameters; they are not estimates derived from our nine-base strings.

## 5. EM: distribute evidence, then update the estimate

The hidden variable is the origin of each ambiguous fragment. **Expectation-maximization (EM)** alternates between estimating those hidden assignments and updating the transcript mixture. We maximize the likelihood, or equivalently minimize its negative logarithm.

For a class e, the E-step computes a responsibility for each compatible transcript:

<div class="k-equation">
  w<sub class="k-sub">e,t</sub> = (α<sub class="k-sub">t</sub> / ℓ<sub class="k-sub">t</sub>) / (∑<sub class="k-sub">j ∈ e</sub> α<sub class="k-sub">j</sub> / ℓ<sub class="k-sub">j</sub>) &nbsp; for t ∈ e<br>
  w<sub class="k-sub">e,t</sub> = 0 &nbsp; for t ∉ e<br>
  n<sub class="k-sub">t</sub> = ∑<sub class="k-sub">e</sub> c<sub class="k-sub">e</sub> w<sub class="k-sub">e,t</sub>
</div>

Responsibilities within a class sum to one. Multiplying them by the class count divides its evidence without duplicating it. Summing those fractional assignments gives expected transcript counts n. The M-step is then:

<div class="k-equation">α<sub class="k-sub">t</sub><sup>new</sup> = n<sub class="k-sub">t</sub> / N, &nbsp; where N = ∑<sub class="k-sub">e</sub> c<sub class="k-sub">e</sub>.</div>

We do **not** divide by effective length again in this M-step: α is a fragment fraction. Confusing it with molecular abundance would change the model.

### A calculation you can reproduce

Take 100 observations unique to T1, 10 unique to T2, and 90 compatible with both. Use equal effective lengths and equal starting weights. T3 has no supporting class here.

The first E-step splits the ninety shared observations 45/45. Expected counts are `(145, 55, 0)`, so the M-step yields `(0.725, 0.275, 0)`. The next E-step allocates `90 × 0.725 = 65.25` to T1 and `24.75` to T2. Updating gives `(0.82625, 0.17375, 0)`.

**Try this:** use the **Worked example**, press E-step, inspect the row allocations, then press M-step. Repeat once to reproduce those numbers. If you loaded your own sample above, choose the worked example to restore these inputs.

{% include kallisto/em-lab.html %}

In this particular example, the shared class cannot distinguish T1 from T2 once their total share is one. At the optimum their ratio is set by the unique evidence: `100:10`, giving approximately `(0.90909, 0.09091, 0)`. EM gets there through repeated fractional reassignment, not random guesses at abundance.

### Why estimated counts and TPM differ

Fractional assignments naturally give noninteger estimated counts. Once EM has fit the fragment shares, the model reports estimated counts `N × α`. Length-normalized expression is then:

<div class="k-equation">
  r<sub class="k-sub">t</sub> = n<sub class="k-sub">t</sub> / ℓ<sub class="k-sub">t</sub><br>
  TPM<sub class="k-sub">t</sub> = 10<sup>6</sup> × r<sub class="k-sub">t</sub> / ∑<sub class="k-sub">j</sub> r<sub class="k-sub">j</sub>
</div>

TPM sums to one million when the total is positive. It is not simply α multiplied by a million unless the effective lengths are equal. In the sandbox table, `N × α` at iteration zero is only an initial guess; after fitting it becomes the estimated fragment count. It is still not a direct count of original RNA molecules.

### Does EM converge in a few steps?

There is no such promise. For this model, with fixed positive lengths, write the log likelihood as:

<div class="k-equation">log L(α) = ∑<sub class="k-sub">e</sub> c<sub class="k-sub">e</sub> log(∑<sub class="k-sub">t ∈ e</sub> α<sub class="k-sub">t</sub> / ℓ<sub class="k-sub">t</sub>) + constant.</div>

Each term is the log of a positive linear function of α, so it is concave. Summing with nonnegative counts preserves concavity, and the allowed mixture vectors form a convex simplex. This gives the basic abundance objective a useful geometry: a local maximum is global, though it need not be unique. This statement is about this fixed model, not every richer RNA-seq model or a jointly trained neural network.

EM does not decrease this objective in exact arithmetic, but improvement can become tiny long before every parameter is determined precisely. Positive initialization also matters: if an EM weight reaches exactly zero, the multiplicative responsibility update cannot revive it.

Select **Slow evidence**. There is only one unique observation and 999 shared ones. Once T3 is removed, the T1 update is `a′ = (1 + 999a) / 1000`. The distance to the optimum shrinks by a factor of 0.999 per iteration. That is a concrete reason why thousands of updates can still be insufficient for a strict tolerance. The widget reports its iteration cap instead of calling that convergence.

### Convergence is not identifiability

Now select **Unidentifiable**. Only `{T1, T2}` appears, and the effective lengths are equal. Once α3 = 0, its likelihood depends on α1 + α2, so every split with that sum equal to one fits equally well.

Run with equal starting weights, then click **Start favoring T1** and run again. You will get different T1/T2 splits with the same final likelihood. More observations of exactly that shared class cannot break the tie. We need distinguishing evidence or extra assumptions.

This matters when reading results: optimizer stability, statistical identifiability, and biological truth are three different claims.

## 6. Bootstrap: uncertainty without another sequencing run

Bootstrapping resamples the existing evidence with replacement. For class counts c and total N, a replicate draws:

<div class="k-equation">c* ∼ Multinomial(N; c<sub class="k-sub">1</sub>/N, …, c<sub class="k-sub">m</sub>/N), &nbsp; then refit EM.</div>

The original method resamples EC counts directly rather than repeating pseudoalignment for every replicate. This follows from their sufficiency for the fitted objective. See the [bootstrap methods](https://arxiv.org/pdf/1505.02710).

**Try this:** run the bootstrap on the worked example. The dots show how estimated fragment shares vary across resamples. Next, repeat with the unidentifiable example and read the warning beneath the plot: a narrow spread can be misleading when the optimizer always picks the same arbitrary solution.

{% include kallisto/bootstrap-lab.html %}

The bootstrap does not create new RNA-seq evidence, reveal absent transcripts, or measure biological variation between people. It explores sampling variability under the fitted procedure and its assumptions.

New sequencing is a separate operation. More reads from the same library must still be processed. In our fixed toy model, matching class counts can be added and the model refit. In a real workflow, the reference and processing must agree, and fragment-length estimates may need updating. New biological samples usually need their own abundance estimates; they are not pooled automatically. The [manual’s sample-handling guidance](https://pachterlab.github.io/kallisto/manual) makes this distinction explicit.

## 7. Would a neural-network surrogate be easier?

It is tempting to learn a function from EC counts directly to abundance. My first question would be what the training target is. If it is EM’s output, we have paid for the original solver and introduced another approximation. If it is simulated truth, the simulator’s assumptions become part of the learned estimator.

The input structure also changes with the transcript reference, and a network cannot reconstruct evidence that the input does not contain. Our unidentifiable example is a useful test: a confident prediction of a particular split must come from a prior or learned pattern, not from the shared counts alone.

A more focused research idea is to learn an initialization or propose accelerated updates, then check them against the likelihood and a convergence criterion. That might help repeated workloads. It remains a hypothesis to benchmark, including training cost, reference changes, rare transcripts, and numerical behavior. For this small fixed likelihood, we already know the update algebra; first I would investigate executing that algebra efficiently.

<h2 id="gpu">8. What changes on a GPU?</h2>

That brings us to [RNA-seq analysis in seconds using GPUs](https://www.biorxiv.org/content/10.64898/2026.03.04.709526v1.full.pdf), the March 2026 preprint by Páll Melsted, Elís Mar Guðnýjarson, and Jóhannes Nordal. Melsted was also an original kallisto author. Their implementation parallelizes pseudoalignment and sparse EM, and uses GPU decompression for BGZF input; ordinary gzip still decompresses on the CPU. [Manuscript text](https://www.researchgate.net/publication/401675838_RNA-seq_analysis_in_seconds_using_GPUs).

The authors report roughly 30× speedup excluding setup in a 100-sample Geuvadis experiment. Their workstation has an RTX 5090 with 32 GB VRAM, a Ryzen 9 9900X, 128 GB RAM, and NVMe storage. Inputs were converted to BGZF; CPU samples ran concurrently with four threads each, while GPU samples ran serially with four CPU threads. A separate 295-million-pair dataset took about 40 minutes on 16 CPU threads versus 50 seconds with the GPU. These are reported configurations, not portable speed guarantees. [Benchmark methods and results](https://www.researchgate.net/publication/401675838_RNA-seq_analysis_in_seconds_using_GPUs).

### See the parallel work in our experiment

The following is my interpretation of the computational structure. Different reads can perform lookup work concurrently; different classes can compute responsibility denominators concurrently. The harder part is arranging data and combining contributions efficiently.

Write a sparse matrix B with one row per class and one column per transcript, where `B[e,t] = 1 / ℓ[t]` for a compatible pair and zero otherwise. Then our EM update can be expressed as:

<div class="k-equation">
  d = Bα<br>
  u = c / d &nbsp; (elementwise, for observed classes)<br>
  n = α ⊙ (B<sup>T</sup>u)<br>
  α<sup>new</sup> = n / N
</div>

This is the same responsibility calculation rearranged into sparse matrix-vector products and elementwise operations. It suggests GPU work without requiring a neural network or replacing the likelihood. It also explains why fast dense matrix multiplication alone is not enough: the compatibility matrix is sparse and irregular, and class sizes vary.

For a tiny sample, launch overhead and data movement may outweigh parallelism. At larger scale, scattered index reads, reductions into shared transcript counters, memory capacity, and imbalance between short and long classes all matter. Batch size and memory layout become algorithmic decisions.

### The bottleneck moves

For the Geuvadis benchmark, the paper reports about 24.1 million read pairs per second for mapping but about 3.6 million pairs per second overall. EM, decompression, transfers, and other work consume the gap. The practical lesson is to measure the complete pipeline. [Runtime breakdown](https://www.researchgate.net/publication/401675838_RNA-seq_analysis_in_seconds_using_GPUs).

If a fraction p of runtime is accelerated by s, with the rest unchanged, total speedup is `1 / ((1 − p) + p/s)`. For example, making an 80% portion infinitely fast still limits overall speedup to 5×. This arithmetic is why storage, compression format, parsing, and output belong in an acceleration experiment.

The preprint supplies evidence about runtime; its speedups do not establish greater biological accuracy. Code is available in the [kallisto GPU branch](https://github.com/pachterlab/kallisto/tree/gpu). Keeping the statistical objective fixed and checking numerical agreement is a different question from comparing competing abundance models.

## 9. TODO: revisit the entire Bowtie2 + RSEM pipeline

The historical tradeoff was compelling: skip detailed alignments, keep compatibility information, and quantify much faster. As hardware progresses, we should reconsider which computations are affordable.

I want to revisit **the whole Bowtie2 + RSEM pipeline on GPUs**, including alignment, the information passed to quantification, and inference. Bowtie2 + RSEM looks like a strong accuracy candidate from the original kallisto comparison, but “the most accurate” must stay a benchmark hypothesis. Accuracy depends on the dataset and reference, and a simulator built around one model can favor it. Later work also shows that [mapping methodology affects abundance accuracy on real data](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-020-02151-8).

A useful experiment would compare GPU kallisto with a GPU-oriented alignment-and-quantification pipeline on identical references and libraries. Measure end-to-end runtime, setup and format-conversion cost, peak memory, and gene- and transcript-level accuracy, with special attention to rare or ambiguous isoforms. Use multiple simulation assumptions and independent experimental evidence. Preserving valid input to RSEM also matters: its [documented alignment requirements](https://github.com/deweylab/RSEM#using-an-alternative-aligner) constrain what an accelerated aligner may emit.

**Leaving this as a TODO:** as hardware improves, reconsider algorithms around GPUs—including all of Bowtie2 + RSEM—and test whether richer alignment information now offers a better accuracy–runtime tradeoff. The next step is to measure that possibility.

<script defer src="{{ site.baseurl }}/assets/kallisto/model.js"></script>
<script defer src="{{ site.baseurl }}/assets/kallisto/playground.js"></script>
