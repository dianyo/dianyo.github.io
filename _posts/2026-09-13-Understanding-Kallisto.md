---
layout: post
title: "Understanding kallisto: an interactive guide from reads to abundance"
permalink: /understanding-kallisto/
excerpt: "Change a read, follow its k-mers, and watch EM estimate transcript abundance. Explore how kallisto works, what its estimates mean, and what GPUs can make faster."
---

RNA sequencing produces millions of short sequences called **reads**. The challenge is to work backward from those reads: which RNA transcripts produced them, and how abundant was each transcript? Many transcripts share sequence, so a read often has several possible origins.

<link rel="stylesheet" href="{{ site.baseurl }}/assets/kallisto/vendor/katex-0.18.7/katex.min.css">
<link rel="stylesheet" href="{{ site.baseurl }}/assets/kallisto/article.css">

**Kallisto answers this question by finding candidate transcripts for each read, grouping reads with the same candidates, and estimating how to divide the evidence between transcripts.** It can do this without calculating a full alignment for every read.

The examples below let you follow that process. Change a sequence to see its candidates change, adjust the number of reads to change the evidence, and step through the abundance calculation. Those same operations will then provide a way to understand what GPUs can accelerate—and why faster hardware might make it worth revisiting more detailed alignment methods.

<div class="k-roadmap" aria-label="Article contents">
  <strong>Read it in order, or jump into an experiment</strong>
  <ol>
    <li><a href="#the-problem">Why can one read have several origins?</a></li>
    <li><a href="#sequence-lab">Play with k-mers and pseudoalignment</a></li>
    <li><a href="#class-lab">Build compatibility classes</a></li>
    <li><a href="#em-lab">Step through EM</a></li>
    <li><a href="#bootstrap-lab">Resample the data</a></li>
    <li><a href="#gpu">What changes on a GPU?</a></li>
  </ol>
</div>

<noscript><p class="k-note">JavaScript is disabled. The article remains readable; equations appear as LaTeX source, and the experiments require JavaScript.</p></noscript>

<h2 id="the-problem">1. The problem: one fragment, several possible origins</h2>

A gene can produce several RNA transcripts through alternative splicing. These versions, called **isoforms**, share some sequence and differ elsewhere. RNA sequencing, usually shortened to **RNA-seq**, samples fragments from the RNA population. A read records the sequence at an end of a fragment and may cover only a region shared by several isoforms.

In paired-end sequencing, both ends of a fragment are read. The two reads provide evidence about the same fragment's origin, so the pair is counted as one observation. The sequence examples below use one read per observation to keep the matching steps easy to follow.

Consider three reference transcripts, labeled T1, T2, and T3. Their sequences are only nine bases long so that every match can be inspected:

```text
T1   ACTGACGTA
T2   ACTGACCTA
T3   GGTGACGTA
```

The read `TGACGTA` occurs in both T1 and T3. An **alignment** describes where a read matches a reference, base by base, including mismatches or gaps. Even a perfect alignment of this read would leave two possible origins. Discarding it would waste evidence; counting it once for T1 and once for T3 would count the same observation twice.

Kallisto first asks a smaller question: **which transcripts are compatible with this read?** Finding that candidate set is called **pseudoalignment**. A statistical model then uses the evidence across all reads to estimate each transcript's contribution. This separation is central to [Bray, Pimentel, Melsted, and Pachter’s original paper](https://www.nature.com/articles/nbt.3519).

<div class="k-flow" aria-label="Algorithm stages"><span>Read sequences</span><span>→ candidate sets</span><span>→ class counts</span><span>→ abundance</span></div>

The reference transcriptome is the collection of transcript sequences supplied to kallisto. Its abundance estimates describe transcripts in that collection; a transcript missing from the reference cannot receive its own estimate.

## 2. K-mers: make sequence searchable

A **k-mer** is a stretch of k consecutive bases. Set k to 3, slide a three-base window along `TGACGTA`, and the read becomes five overlapping windows:

```text
TGA
 GAC
  ACG
   CGT
    GTA
```

A read of length <span class="k-math" markdown="0">\(L\)</span> has <span class="k-math" markdown="0">\(L-k+1\)</span> such windows when <span class="k-math" markdown="0">\(L\geq k\)</span>. An **index** makes the windows searchable: for each distinct k-mer, it records which reference transcripts contain it. Here, `TGA` occurs in all three transcripts, while `ACG` occurs in T1 and T3. A transcript either belongs to that candidate set or does not; repeated occurrences within it do not add extra transcript identities.

**Try it:** leave k at 3 and select **Shared read**. Click `TGA`, then `ACG`. TGA leaves all three transcripts possible; ACG removes T2. Next, select **Unique read** and follow its windows. Its beginning is shared with T2 and its end is shared with T3, but only T1 survives both pieces of evidence.

{% include kallisto/sequence-lab.html %}

The operation is called **set intersection**: keep only candidates that appear in every matching window's set. In the notation below, <span class="k-math" markdown="0">\(C\)</span> means “candidate set,” braces enclose its members, and <span class="k-math" markdown="0">\(\cap\)</span> means “intersect”:

<div class="k-equation">
\[
\begin{aligned}
C(\mathtt{TGACGTA})
  &= C(\mathtt{TGA}) \cap C(\mathtt{GAC}) \\
  &\quad \cap C(\mathtt{ACG}) \cap C(\mathtt{CGT}) \cap C(\mathtt{GTA}) \\
  &= \{T_1,T_2,T_3\} \cap \{T_1,T_3\} \\
  &= \{T_1,T_3\}.
\end{aligned}
\]
</div>

The final answer <span class="k-math" markdown="0">\(\{T_1,T_3\}\)</span> means either transcript could be the origin. It does not yet assign a probability to either one. Also notice that the five windows still represent **one read**. They help determine its candidates; they do not become five separate observations in the abundance calculation.

### What changing k teaches us

Small k-mers occur more readily by chance. Longer k-mers can distinguish sequences better, but give a short read fewer windows. A single changed base also affects every window that overlaps it.

**Try it:** select **One substitution** at k = 3. A substitution replaces one base with another. Some windows now fail to match, while unaffected windows can still identify a candidate. Increase k and watch how many matching windows remain. Then select **No matches** to see what happens when none of the windows supplies evidence.

An absent k-mer is skipped in this intersection procedure. If none of the windows matches, the read is **unassigned**. A read is also unassigned when its matching windows point to incompatible sets with no transcript in common. A changed base can sometimes create a match elsewhere in the reference, so surviving matches are evidence rather than a guarantee of the true origin.

For these short examples, k ranges from 2 to 7 and sequences are compared in their written orientation. Real RNA-seq requires attention to read orientation and paired ends. Kallisto's documented index default is k = 31, with an odd-k requirement; those settings are described in the [kallisto manual](https://pachterlab.github.io/kallisto/manual).

### Where the graph fits

Notice that several consecutive windows can have identical candidate sets. Once <span class="k-math" markdown="0">\(\{T_1,T_3\}\)</span> is the running answer, intersecting it with <span class="k-math" markdown="0">\(\{T_1,T_3\}\)</span> again changes nothing. Avoiding redundant work is one source of kallisto's speed.

The original index organizes k-mers into a **transcriptome de Bruijn graph**. K-mers are nodes, neighboring sequence windows connect them, and transcript membership supplies their “colors.” Each transcript follows a path through the graph. Linear stretches with unchanged membership can be compacted into **contigs**, which may cover only part of a transcript. Kallisto uses this structure to skip redundant lookups and checks the end of a skip. The example above exposes every window so you can see which intersections change the answer. See the [original indexing and pseudoalignment methods](https://arxiv.org/pdf/1505.02710).

There is a limit to what the candidate sets say. They do not retain the order and positions of every match. With very small k, a string can pass the intersection test even when it does not occur as one continuous sequence in a transcript. This is one reason the choice of k matters. For another walkthrough of the core algorithm, see [IRIC's explanation of how kallisto works](https://bioinfo.iric.ca/understanding-how-kallisto-works/).

## 3. Compatibility classes: group reads with the same candidates

After pseudoalignment, many reads have the same candidate set. Grouping them makes the next calculation smaller. Three terms describe the successive stages:

- A **k-mer compatibility set** contains transcripts that contain that k-mer.
- A **read or fragment compatibility set** contains candidates surviving the combined evidence.
- An **equivalence class (EC)** groups observations with the same final candidate set. Its count is the number of observations in that group. These are also called transcript compatibility counts (TCCs).

For example, <span class="k-math" markdown="0">\(\{T_1,T_2\}:90\)</span> means ninety fragments could have come from T1 or T2. Their origins are still unresolved. The grouping records one count of ninety, which the abundance model will divide between candidates.

**Try it:** change the copies of `ACTGACG` from 100 to 200. At k = 3, this adds evidence to the <span class="k-math" markdown="0">\(\{T_1\}\)</span> class. Then change k in experiment 1 and watch the same sample regroup. Adding a read with no matches increases the unassigned total. Press **Use these class counts in EM** to send the assigned counts to the abundance experiment in section 5.

{% include kallisto/class-lab.html %}

### What does grouping preserve?

Consider a compatibility matrix—a table with a row for each read and a column for each transcript. A 1 means the transcript remains a candidate; a 0 means it does not:

<div class="k-equation">
\[
\begin{array}{c|ccc}
 & T_1 & T_2 & T_3 \\ \hline
\text{read A} & 1 & 1 & 0 \\
\text{read B} & 1 & 1 & 0 \\
\text{read C} & 1 & 0 & 0 \\
\text{read D} & 1 & 1 & 0
\end{array}
\]
</div>

The grouped version is <span class="k-math" markdown="0">\(\{T_1,T_2\}:3\)</span> and <span class="k-math" markdown="0">\(\{T_1\}:1\)</span>. It preserves how often each candidate set occurs, but not which row belonged to read A, B, C, or D. You cannot reconstruct the original labeled table from the counts alone.

For abundance estimation, however, those repeated rows ask the model exactly the same question. A **likelihood** measures how well a proposed mixture of transcripts explains the observed evidence. Each repeated row contributes the same factor to that likelihood, so multiplying it three times is equivalent to raising it to the third power.

In symbols, let <span class="k-math" markdown="0">\(\alpha_t\)</span> be the probability that a sampled fragment comes from transcript t, and <span class="k-math" markdown="0">\(\ell_t\)</span> its effective length, explained next. Let e denote a candidate set and <span class="k-math" markdown="0">\(c_e\)</span> its count. The symbols <span class="k-math" markdown="0">\(\sum\)</span> and <span class="k-math" markdown="0">\(\prod\)</span> mean “sum” and “multiply,” respectively. Here <span class="k-math" markdown="0">\(F\)</span> contains the assigned fragments and <span class="k-math" markdown="0">\(E\)</span> contains their equivalence classes. Kallisto's basic likelihood can be written as:

<div class="k-equation">
\[
\begin{aligned}
g_e(\boldsymbol{\alpha}) &= \sum_{t\in e}\frac{\alpha_t}{\ell_t}, \\
\mathcal{L}(\boldsymbol{\alpha}) &\propto \prod_{f\in F}g_{C(f)}(\boldsymbol{\alpha}) \\
  &= \prod_{e\in E}\bigl[g_e(\boldsymbol{\alpha})\bigr]^{c_e}, \\
\alpha_t &\geq 0, \qquad \sum_t\alpha_t=1.
\end{aligned}
\]
</div>

The symbol <span class="k-math" markdown="0">\(\propto\)</span> omits factors independent of the proposed abundance. For the four rows above, both representations give <span class="k-math" markdown="0">\(g_{12}\,g_{12}\,g_1\,g_{12}=g_{12}^{3}g_1\)</span>. The equality is ordinary factorization, with no numerical approximation. The counts are called **sufficient statistics** for this fixed likelihood because they retain everything needed to calculate it. See the [original quantification methods](https://arxiv.org/pdf/1505.02710).

The distinction matters: reducing sequence evidence to candidate sets can discard information, while grouping identical rows preserves this particular likelihood exactly. A richer model might use alignment scores or give different weights to fragments with the same candidates. For such a model, the counts alone might no longer be sufficient.

## 4. Why effective length appears

Imagine two transcripts present in equal numbers of RNA molecules, but one offers twice as many possible fragment starts. It can contribute more fragments simply because it is longer. **Fragment share** <span class="k-math" markdown="0">\(\alpha\)</span> therefore differs from the transcript's share of the RNA molecules.

**Effective length** accounts for the available fragment starts. For a transcript of length <span class="k-math" markdown="0">\(L\)</span> and a fixed fragment length <span class="k-math" markdown="0">\(d\leq L\)</span>, a simple effective length is <span class="k-math" markdown="0">\(\ell=L-d+1\)</span>. A 1,000-base transcript with 200-base fragments, for example, has 801 possible starts. Real library models account for a distribution of fragment lengths. For background, see Pachter’s [Models for transcript quantification from RNA-Seq](https://arxiv.org/abs/1104.3889).

The basic model describes two choices: select transcript t with probability <span class="k-math" markdown="0">\(\alpha_t\)</span>, then select one of its possible fragment locations. The second choice contributes the inverse-length factor. That is why a compatible transcript contributes <span class="k-math" markdown="0">\(\alpha/\ell\)</span> to the likelihood term.

With equal fragment shares but effective lengths 100 and 200, a shared observation has relative weights <span class="k-math" markdown="0">\(\frac{0.5}{100}:\frac{0.5}{200}=2:1\)</span>. You can explore this by changing effective lengths in the EM controls. Those lengths are independent parameters for the abundance examples, separate from the nine-base sequences used to explore matching.

## 5. EM: distribute evidence, then update the estimate

Suppose unique reads strongly support T1, but only weakly support T2. It would be surprising to divide all reads shared by T1 and T2 equally. Evidence from the whole sample should influence that division.

**Expectation-maximization (EM)** does this in two repeating steps. The **E-step** uses the current abundance estimate to divide ambiguous evidence. The **M-step** adds up those assignments and uses the totals as the next abundance estimate. The origin of each ambiguous fragment is the hidden information being estimated.

For a class e, the fraction allocated to a compatible transcript is called its **responsibility**, written <span class="k-math" markdown="0">\(w_{e,t}\)</span>. It is that transcript's <span class="k-math" markdown="0">\(\alpha/\ell\)</span> weight divided by the total weight of the candidates:

<div class="k-equation">
\[
\begin{aligned}
w_{e,t} &= \begin{cases}
\displaystyle\frac{\alpha_t/\ell_t}{\sum_{j\in e}\alpha_j/\ell_j}, & t\in e, \\[8pt]
0, & t\notin e,
\end{cases} \\
n_t &= \sum_{e\in E}c_e\,w_{e,t}.
\end{aligned}
\]
</div>

Responsibilities within a class sum to one. Multiplying them by the class count divides its evidence without duplicating it. Summing those fractional assignments gives expected transcript counts <span class="k-math" markdown="0">\(n_t\)</span>. The M-step is then:

<div class="k-equation">
\[
\alpha_t^{\mathrm{new}}=\frac{n_t}{N},
\qquad N=\sum_{e\in E}c_e.
\]
</div>

Effective length already entered the responsibilities. The M-step simply divides expected counts by <span class="k-math" markdown="0">\(N\)</span>; <span class="k-math" markdown="0">\(\alpha\)</span> remains a fragment fraction.

### A calculation you can reproduce

Take 100 observations unique to T1, 10 unique to T2, and 90 compatible with both. Use equal effective lengths and equal starting weights. T3 has no supporting class here.

The first E-step splits the ninety shared observations 45/45. Expected counts are <span class="k-math" markdown="0">\((145,55,0)\)</span>, so the M-step yields <span class="k-math" markdown="0">\((0.725,0.275,0)\)</span>. The next E-step allocates <span class="k-math" markdown="0">\(90\times0.725=65.25\)</span> to T1 and 24.75 to T2. Updating gives <span class="k-math" markdown="0">\((0.82625,0.17375,0)\)</span>.

**Try it:** select **Worked example**, press **E-step: split counts**, and find the row containing the ninety shared fragments. After checking the 45/45 split, press **M-step: update α**. Repeat once to reproduce <span class="k-math" markdown="0">\((0.82625,0.17375,0)\)</span>. Then edit the unique counts or use your sample from section 3 and see how the shared evidence is redistributed.

{% include kallisto/em-lab.html %}

In this example, the final T1:T2 ratio is set by the unique evidence: <span class="k-math" markdown="0">\(100:10\)</span>. Running EM to convergence gives approximately <span class="k-math" markdown="0">\((0.90909,0.09091,0)\)</span>. The ninety shared fragments add to the estimated counts but cannot distinguish those two transcripts by themselves.

### Why estimated counts and TPM differ

Fractional assignments naturally give noninteger estimated counts. Once EM has fit the fragment shares, the estimated count for a transcript is <span class="k-math" markdown="0">\(N\alpha\)</span>. To account for transcript length, divide these counts by effective length and scale the resulting shares to a total of one million. The result is **TPM**, or transcripts per million:

<div class="k-equation">
\[
\begin{aligned}
r_t &= \frac{n_t}{\ell_t}, \\
\mathrm{TPM}_t &= 10^6\frac{r_t}{\sum_j r_j}.
\end{aligned}
\]
</div>

TPM sums to one million when the total is positive. It equals <span class="k-math" markdown="0">\(\alpha\)</span> multiplied by a million only when the effective lengths are equal. Compare the <span class="k-math" markdown="0">\(\alpha\)</span> and TPM columns after changing the lengths: they answer different questions about the same sample. At iteration zero, <span class="k-math" markdown="0">\(N\alpha\)</span> is only an initial guess; after fitting, it is an estimated fragment count rather than a direct count of original RNA molecules.

### How quickly does EM converge?

EM aims to maximize the likelihood. For calculation, it is convenient to take the logarithm, which turns the product of many terms into a sum without changing where the maximum occurs. With fixed positive lengths:

<div class="k-equation">
\[
\log\mathcal{L}(\boldsymbol{\alpha})
=\sum_{e\in E}c_e\log\!\left(\sum_{t\in e}\frac{\alpha_t}{\ell_t}\right)
+\mathrm{constant}.
\]
</div>

This basic objective has a useful property: a local maximum is also a global maximum, although several different abundance estimates can share that maximum. Mathematically, the log likelihood is **concave**: each term is the log of a positive linear function of <span class="k-math" markdown="0">\(\alpha\)</span>, and the terms are summed with nonnegative counts. The allowed abundance vectors—nonnegative shares summing to one—form a convex set. These properties apply to this fixed model; richer models need their own analysis.

Each EM iteration does not decrease the likelihood in exact arithmetic. That does not guarantee a quick answer: many small improvements may be needed. Starting with positive weights also matters, because a transcript given exactly zero abundance receives zero responsibility and cannot recover through these updates.

**Try it:** select **Slow evidence** and run EM. There is only one unique observation and 999 shared ones. Once T3 is removed, the T1 update is <span class="k-math" markdown="0">\(a'=\frac{1+999a}{1000}\)</span>. Each iteration removes only one thousandth of the remaining distance to the optimum. Even 2,000 updates fail to reach the example's stopping tolerance. This is slow convergence caused by weak distinguishing evidence.

### Can a stable answer still be ambiguous?

Select **Unidentifiable**. Only <span class="k-math" markdown="0">\(\{T_1,T_2\}\)</span> appears, and the effective lengths are equal. No observation distinguishes the two transcripts. Once <span class="k-math" markdown="0">\(\alpha_3=0\)</span>, the likelihood depends on <span class="k-math" markdown="0">\(\alpha_1+\alpha_2\)</span>, so every split with that sum equal to one fits equally well.

**Try it:** run with equal starting weights, then click **Start favoring T1** and run again. The T1/T2 splits differ, but the final likelihood is the same. The starting weights selected a split that the observations themselves could not determine.

This is a lack of **identifiability**: the evidence does not determine a unique answer. More reads from exactly the same shared class cannot resolve it; distinguishing reads or additional assumptions are needed. A stable optimizer is therefore only one part of interpreting an abundance estimate.

## 6. Bootstrap: uncertainty without another sequencing run

How much might an estimate change because a different collection of fragments happened to be sampled? **Bootstrapping** explores that question using the observations already available.

Imagine putting the <span class="k-math" markdown="0">\(N\)</span> observed fragments on cards. Draw a card, record it, put it back, and repeat until you have <span class="k-math" markdown="0">\(N\)</span> draws. Some original observations will appear several times; others will not appear at all. Re-estimate abundance from that resampled collection, then repeat the process to obtain a distribution of estimates.

Because the model uses class counts, the same resampling can be done directly over classes. A class containing one quarter of the observations has a one-quarter chance of being selected on each draw. In statistical notation:

<div class="k-equation">
\[
\mathbf{c}^{*}\sim\operatorname{Multinomial}\!\left(
N;\frac{c_1}{N},\ldots,\frac{c_m}{N}\right).
\]
</div>

Here <span class="k-math" markdown="0">\(\mathbf{c}^{*}\)</span> denotes the new counts, and the **multinomial distribution** describes how <span class="k-math" markdown="0">\(N\)</span> draws are divided between classes. Directly resampling these counts avoids repeating pseudoalignment for each replicate. See the [original bootstrap methods](https://arxiv.org/pdf/1505.02710).

**Try it:** choose **Worked example** in the EM experiment, then run the bootstrap below. Each dot is one estimate from a resampled collection. Repeat with **Unidentifiable**: the dots can collapse onto the same answer because every fit starts from the same weights. That narrow spread does not mean the evidence has resolved T1 from T2.

{% include kallisto/bootstrap-lab.html %}

The spread describes sampling variability under the model and fitting procedure. It does not add biological evidence or measure variation between people. Systematic problems, such as a missing reference transcript, are not repaired by resampling the same data.

Collecting more reads from the same library adds actual observations. They still need to be processed, and combining their class counts requires consistent reference and model settings; fragment-length estimates may also need updating. A new biological sample usually gets its own abundance estimate. The [manual's sample-handling guidance](https://pachterlab.github.io/kallisto/manual) describes how to handle samples spread across several input files.

## 7. Could a neural network replace EM?

A neural network could be trained to predict abundance directly from class counts. Such a **surrogate** approximates a calculation with a learned function. Whether it is useful depends on what supplies its training targets and how often the trained model will be reused. Learning from EM outputs requires running EM to generate those targets; learning from simulated truth carries the simulator's assumptions into the predictions.

Changing the transcript reference also changes the possible classes and outputs. More fundamentally, a network cannot recover a uniquely determined origin from evidence that never distinguished the candidates. In the unidentifiable example, a network's preferred split would reflect learned patterns or other assumptions beyond the shared counts.

A possible research direction is to learn better starting weights or propose faster updates, then check the results against the likelihood and a convergence criterion. Any benefit would need to account for training cost and behavior on new references or rare transcripts.

There is also a direct route to acceleration: execute the existing calculations faster. The experiments have exposed the work—many sequence lookups, intersections, and weighted sums. How much of that work can happen at the same time?

<h2 id="gpu">8. What changes on a GPU?</h2>

GPUs can perform many operations concurrently. Kallisto offers several opportunities: look up windows from different reads, intersect candidate sets for different fragments, and calculate assignments for different classes. The challenge is keeping the GPU supplied with useful work and combining the results efficiently.

[RNA-seq analysis in seconds using GPUs](https://www.biorxiv.org/content/10.64898/2026.03.04.709526v1.full.pdf), a March 2026 preprint by Páll Melsted, Elís Mar Guðnýjarson, and Jóhannes Nordal, explores this approach. Melsted also coauthored the original kallisto paper. The work accelerates pseudoalignment and EM, and uses GPU decompression for BGZF, a format containing independently decompressible blocks. Ordinary gzip input still decompresses on the CPU. [Manuscript text](https://www.researchgate.net/publication/401675838_RNA-seq_analysis_in_seconds_using_GPUs).

The authors report roughly 30× speedup, excluding setup, across 100 Geuvadis samples. Their original benchmark plot below shows how runtime grows with sample size: blue points are CPU kallisto runs, and red points are GPU runs.

<figure class="k-paper-figure k-paper-benchmark" id="gpu-benchmark">
  <a href="{{ site.baseurl }}/images/kallisto/figure-1-benchmark.jpg" aria-label="Open the original benchmark plot at full size">
    <img src="{{ site.baseurl }}/images/kallisto/figure-1-benchmark.jpg" width="1280" height="1251" loading="lazy" alt="Wall time versus sample read count for 100 Geuvadis samples. CPU kallisto times rise from about 100 to 380 seconds; GPU kallisto times remain below about 20 seconds across the plotted range.">
  </a>
  <figcaption>Figure 1 from <a href="https://doi.org/10.64898/2026.03.04.709526">Melsted, Guðnýjarson, and Nordal (2026)</a>, reproduced unchanged under <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>. Select the image to enlarge it.</figcaption>
</figure>

The benchmark used an RTX 5090 with 32 GB GPU memory, a Ryzen 9 9900X, 128 GB system memory, and NVMe storage. Inputs were converted to BGZF; CPU samples ran concurrently with four threads each, while GPU samples ran one at a time with four CPU threads. A separate dataset of 295 million read pairs took about 40 minutes on 16 CPU threads versus 50 seconds with the GPU. Those hardware, input-format, and scheduling details matter when interpreting the comparison.

### From k-mers to GPU arrays

The paper's pipeline diagram connects directly to the sequence experiment. In panel E, overlapping k-mers lead to repeated candidate sets. Deduplication removes repeated sets before intersection, because intersecting a set with itself adds no information. The surviving transcript set is then looked up to identify its class. Organizing these stages as arrays in GPU memory makes it possible to process many reads together.

<figure class="k-paper-figure" id="gpu-pipeline">
  <a href="{{ site.baseurl }}/images/kallisto/figure-2-pipeline.jpg" aria-label="Open the original GPU pipeline diagram at full size">
    <img src="{{ site.baseurl }}/images/kallisto/figure-2-pipeline.jpg" width="1280" height="621" loading="lazy" alt="Panels A to D connect three transcripts, their colored de Bruijn graph, read k-mer matches, and transcript sets. Panel E follows GPU arrays through k-mer lookup, deduplication, transcript-set intersection, and reverse lookup of the resulting class.">
  </a>
  <figcaption>Figure 2 from <a href="https://doi.org/10.64898/2026.03.04.709526">Melsted et al. (2026)</a>, reproduced unchanged under <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>. Select the image to inspect the individual stages.</figcaption>
</figure>

### From fractional assignments to parallel sums

Return to the E-step table. Each row can calculate its own denominator and fractional assignments independently. The column totals then combine contributions from many rows. This provides a way to reason about parallelizing EM directly from its equations.

For a compact mathematical view, define a matrix <span class="k-math" markdown="0">\(B\)</span> with one row per class and one column per transcript. Set <span class="k-math" markdown="0">\(B_{e,t}=1/\ell_t\)</span> for a compatible pair and zero otherwise. Most entries are zero, so <span class="k-math" markdown="0">\(B\)</span> is **sparse**. The update derived earlier can then be rearranged as:

<div class="k-equation">
\[
\begin{aligned}
\mathbf{d} &= B\boldsymbol{\alpha}, \\
\mathbf{u} &= \mathbf{c}\oslash\mathbf{d}, \\
\mathbf{n} &= \boldsymbol{\alpha}\odot\left(B^{\mathsf T}\mathbf{u}\right), \\
\boldsymbol{\alpha}^{\mathrm{new}} &= \frac{\mathbf{n}}{N}.
\end{aligned}
\]
</div>

The first line calculates each class's total candidate weight. The next two lines distribute its count and sum the contributions to each transcript; <span class="k-math" markdown="0">\(\oslash\)</span> means dividing corresponding entries, <span class="k-math" markdown="0">\(\odot\)</span> means multiplying them, and <span class="k-math" markdown="0">\(B^{\mathsf T}\)</span> exchanges rows and columns. Only observed classes participate in these sums, so their denominators are positive when starting weights are positive. The final line normalizes the counts. This is the same EM calculation expressed as sparse matrix-vector products and elementwise operations.

That structure suggests GPU work, but it does not turn the problem into ordinary dense matrix multiplication. Classes contain different numbers of candidates, and contributions from many classes may need to update the same transcript total. Efficient memory access and reductions—the combining of many values into a sum—matter as much as arithmetic throughput.

For a small sample, starting GPU work and moving data may cost more time than parallel execution saves. For a large sample, the index must fit in memory, work must be divided into useful batches, and groups with different amounts of work must be scheduled efficiently. The best organization of an algorithm can therefore depend on the hardware running it.

### The bottleneck moves

For the Geuvadis benchmark, the paper reports about 24.1 million read pairs per second for mapping but about 3.6 million pairs per second overall. EM, decompression, transfers, and other work consume the gap. Its original runtime table makes that shift visible: the GPU EM calculation takes 3,148 ms on average, compared with 722 ms for GPU mapping.

<figure class="k-paper-figure" id="gpu-runtime">
  <a href="{{ site.baseurl }}/images/kallisto/table-1-runtime.png" aria-label="Open the original runtime table at full size">
    <img src="{{ site.baseurl }}/images/kallisto/table-1-runtime.png" width="2190" height="460" loading="lazy" alt="Table 1: average component runtimes across 100 Geuvadis samples. CPU and GPU times respectively: index setup 376 and 933 ms; I/O and decompression 2370 and 841 ms; GPU mapping 514 and 722 ms; EM has no CPU time listed and 3148 ms GPU time.">
  </a>
  <figcaption>Table 1 from <a href="https://doi.org/10.64898/2026.03.04.709526">Melsted et al. (2026)</a>, cropped from PDF page 3 with its content unchanged; <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>. The CPU and GPU columns report work within the GPU implementation, rather than two separate implementations. Select the image to read the full-size table.</figcaption>
</figure>

The practical lesson is to measure the complete pipeline. A fast mapping stage can leave abundance estimation or input processing as the dominant cost.

Suppose a program spends 80 seconds matching reads and 20 seconds doing everything else. Even if matching became instantaneous, the run would still take 20 seconds: a maximum improvement of 5×. Speeding up a stage makes the remaining stages more visible.

More generally, if a fraction <span class="k-math" markdown="0">\(p\)</span> of runtime is accelerated by a factor <span class="k-math" markdown="0">\(s\)</span> while the rest stays unchanged, total speedup is <span class="k-math" markdown="0">\(\frac{1}{(1-p)+p/s}\)</span>. Storage, decompression, parsing the input, transferring data, and writing output all belong in an end-to-end comparison.

The runtime improvements address how quickly the calculation runs. Establishing biological accuracy requires a separate comparison against suitable evidence. The [kallisto GPU branch](https://github.com/pachterlab/kallisto/tree/gpu) provides the code; numerical agreement with the CPU calculation and accuracy relative to other abundance methods are distinct questions.

## 9. TODO: revisit the entire Bowtie2 + RSEM pipeline

Kallisto's central tradeoff is now visible: retain candidate-transcript information, group repeated evidence, and estimate abundance without computing every alignment detail. The examples show both the speed opportunities and the information limits of that approach.

As hardware improves, a more detailed calculation may become affordable. **Bowtie2 + RSEM** is a natural candidate to revisit: Bowtie2 computes alignments, and RSEM uses alignment evidence to estimate expression. Moving the whole pipeline toward GPUs would require considering alignment, the information passed between stages, and abundance estimation together.

Bowtie2 + RSEM looks like a strong accuracy candidate in the original kallisto comparison. Whether it is the most accurate remains a question to test across datasets and references; simulations built around one model can favor that model. Later work also shows that [mapping methodology affects abundance accuracy on real data](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-020-02151-8).

A useful experiment would compare GPU kallisto with an accelerated Bowtie2 + RSEM pipeline using the same references and libraries. It should measure complete runtime, setup and format-conversion costs, peak memory, and gene- and transcript-level accuracy, especially for rare or ambiguous isoforms. Multiple simulation assumptions and independent experimental evidence would help separate general improvements from advantages specific to one benchmark. RSEM's [alignment requirements](https://github.com/deweylab/RSEM#using-an-alternative-aligner) would also need to be preserved.

**TODO:** reconsider the full Bowtie2 + RSEM pipeline on modern GPUs and test whether retaining richer alignment information can deliver a better accuracy–runtime tradeoff. Hardware progress is a reason to revisit the algorithm as a whole.

<script defer src="{{ site.baseurl }}/assets/kallisto/vendor/katex-0.18.7/katex.min.js"></script>
<script defer src="{{ site.baseurl }}/assets/kallisto/math.js"></script>
<script defer src="{{ site.baseurl }}/assets/kallisto/model.js"></script>
<script defer src="{{ site.baseurl }}/assets/kallisto/playground.js"></script>
