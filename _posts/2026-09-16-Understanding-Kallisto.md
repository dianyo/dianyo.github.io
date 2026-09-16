---
layout: post
title: "Understanding kallisto: an interactive guide from reads to abundance"
permalink: /understanding-kallisto/
excerpt: "Change a RNA-seq read, follow its k-mers, and watch EM estimate transcript abundance by kallisto. Explore how kallisto works, what its estimates mean, and what GPUs can make faster."
---

RNA sequencing produces millions of short sequences called **reads**. The challenge is to work backward from those reads to find out which RNA transcripts produced them, and how abundant was each transcript? Many transcripts share sequence, so a read often has several possible origins.

<link rel="stylesheet" href="{{ site.baseurl }}/assets/kallisto/vendor/katex-0.18.7/katex.min.css">
<link rel="stylesheet" href="{{ site.baseurl }}/assets/kallisto/article.css">

**Kallisto estimates the abundance of each reference transcript from the collective evidence in RNA-seq reads. Using its transcriptome index, it pseudoaligns each fragment first: it determines which transcripts are compatible with the fragment without computing exact alignment coordinates. Kallisto then groups fragments with the same compatibility set into equivalence classes and uses Expectation-Maximum (EM) algorithm to estimate transcript abundances jointly across the sample.**

The examples below let you follow that process. You can change a sequence to see its candidates change, adjust the number of reads to change the evidence, and step through the abundance calculation. After that we'll explore what GPUs can accelerate, and why faster hardware might make it worth revisiting more detailed but slower alignment methods (Like Bowtie2 + RSEM).

<div class="k-roadmap" aria-label="Article contents">
  <strong>Roadmap for this article</strong>
  <ol>
    <li><a href="#the-problem">The problem: one fragment, several possible origins</a></li>
    <li>
      <a href="#k-mers">K-mers: make sequence searchable</a>
      <ul>
        <li><a href="#sequence-lab">Experiment 1: Follow a read, one k-mer at a time</a></li>
      </ul>
    </li>
    <li>
      <a href="#compatibility-classes">Compatibility classes: group reads with the same candidates</a>
      <ul>
        <li><a href="#class-lab">Experiment 2: Build your own little RNA-seq sample</a></li>
      </ul>
    </li>
    <li><a href="#effective-length">Why effective length appears</a></li>
    <li>
      <a href="#em">EM: distribute evidence, then update the estimate</a>
      <ul>
        <li><a href="#em-lab">Experiment 3: Be the EM algorithm</a></li>
      </ul>
    </li>
    <li><a href="#neural-network">Could a neural network replace EM?</a></li>
    <li><a href="#gpu">What changes on a GPU?</a></li>
    <li><a href="#bowtie2-rsem">Research direction: revisit Bowtie2 + RSEM on GPUs</a></li>
    <li><a href="#conclusion">Conclusion: two research directions</a></li>
    <li><a href="#references">References</a></li>
  </ol>
</div>

<noscript><p class="k-note">JavaScript is disabled. The article remains readable; equations appear as LaTeX source, and the experiments require JavaScript.</p></noscript>

<h2 id="the-problem">1. The problem: one fragment, several possible origins</h2>

A gene can produce several RNA transcripts through **alternative splicing**. These versions, called **isoforms**, share some sequence and differ elsewhere. RNA sequencing, usually shortened to **RNA-seq**, samples fragments from the RNA population. A read records the sequence at an end of a fragment and may cover only a region shared by several isoforms.

In paired-end sequencing, both ends of a fragment are read. The two reads provide evidence about the same fragment's origin, so the pair is counted as one observation. The sequence examples below use one read per observation to keep the matching steps easy to follow.

Consider three reference transcripts, labeled T1, T2, and T3. Their sequences are only nine bases long so that every match can be inspected:

```text
T1   ACTGACGTA
T2   ACTGACCTA
T3   GGTGACGTA
```

The read `TGACGTA` occurs in both T1 and T3. An **alignment** describes where a read matches a reference, base by base, including mismatches or gaps. Even a perfect alignment of this read would leave two possible origins. Discarding it would waste evidence; counting it once for T1 and once for T3 would count the same observation twice.

Instead of aligning the read to the reference transcripts directly, Kallisto first asks a smaller question: **which transcripts are compatible with this read?** Finding that candidate set is called **pseudoalignment**. A statistical model then uses the evidence across all reads to estimate each transcript's contribution. This separation is central to [Bray, Pimentel, Melsted, and Pachter’s original paper](https://www.nature.com/articles/nbt.3519).

<div class="k-flow" aria-label="Algorithm stages"><span>Read sequences</span><span>→ candidate sets</span><span>→ class counts</span><span>→ abundance</span></div>

The reference transcriptome is the collection of transcript sequences supplied to kallisto. Its abundance estimates describe transcripts in that collection; a transcript missing from the reference cannot receive its own estimate.

<h2 id="k-mers">2. K-mers: make sequence searchable</h2>

A **k-mer** is a stretch of k consecutive bases. Set k to 3, slide a three-base window along `TGACGTA`, and the read becomes five overlapping windows:

```text
TGA
 GAC
  ACG
   CGT
    GTA
```

A read of length <span class="k-math" markdown="0">\(L\)</span> has <span class="k-math" markdown="0">\(L-k+1\)</span> such windows when <span class="k-math" markdown="0">\(L\geq k\)</span>. An **index** makes the windows searchable: for each distinct k-mer, it records which reference transcripts contain it. Here, `TGA` occurs in all three transcripts, while `ACG` occurs in T1 and T3. A transcript either belongs to that candidate set or does not; repeated occurrences within it do not add extra transcript identities.

<aside class="k-try" aria-label="Try it" markdown="1">
**Try it:** leave k at 3 and select **Shared read**. Click `TGA`, then `ACG`. TGA leaves all three transcripts possible; ACG removes T2. Next, select **Unique read** and follow its windows. Its beginning is shared with T2 and its end is shared with T3, but only T1 survives both pieces of evidence.
</aside>

{% include kallisto/sequence-lab.html %}

The operation we used above is called **set intersection**: keep only candidates that appear in every matching k-mer window's set, and <span class="k-math" markdown="0">\(C\)</span> is called **candidate set**:

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

The final answer <span class="k-math" markdown="0">\(\{T_1,T_3\}\)</span> means either transcript could be the origin. It does not yet assign a probability to either one. Also notice that the five windows still represent **one read**. They help determine its candidates only, **they do not become five separate observations in the abundance calculation**.

### What changing k teaches us

Small k-mers occur more readily by chance. Longer k-mers can distinguish sequences better, but give a short read fewer windows. A single changed base also affects every window that overlaps it.

<aside class="k-try" aria-label="Try it" markdown="1">
**Try it:** In the above <a href="#sequence-lab">interactive interface </a> select **One substitution** at k = 3. A substitution replaces one base with another. Some windows now fail to match, while unaffected windows can still identify a candidate. Increase k and watch how many matching windows remain. Then select **No matches** to see what happens when none of the windows supplies evidence.
</aside>

An **absent k-mer is skipped (instead of intersecting with a null set)** in this intersection procedure. If none of the windows matches in the end, the read is **unassigned**. A read is also unassigned when its matching windows point to incompatible sets with no transcript in common. A changed base can sometimes create a match elsewhere in the reference, so surviving matches are evidence rather than a guarantee of the true origin.

For these short examples, k ranges from 2 to 7 and sequences are compared in their written orientation. Real RNA-seq requires attention to read orientation and paired ends. Kallisto's documented index default is k = 31, with an odd-k requirement; those settings are described in the [kallisto manual](https://pachterlab.github.io/kallisto/manual).

### Where the graph fits

Notice that several consecutive windows can have identical candidate sets. Once <span class="k-math" markdown="0">\(\{T_1,T_3\}\)</span> is the running answer, intersecting it with <span class="k-math" markdown="0">\(\{T_1,T_3\}\)</span> again changes nothing. **Avoiding redundant work is one source of kallisto's speed.**

The original index organizes k-mers into a **transcriptome de Bruijn graph (T-DBG)**. K-mers are nodes, neighboring sequence windows connect them, and transcript membership supplies their “colors.” Each transcript follows a path through the graph. Linear stretches with unchanged membership can be compacted into **contigs**, which may cover only part of a transcript. Kallisto uses this structure to skip redundant lookups and checks the end of a skip.

<figure class="k-paper-figure k-paper-graph" id="kallisto-graph-overview">
  <a href="{{ site.baseurl }}/images/kallisto/figure-0-kallisto.jpg" aria-label="Open the original kallisto graph overview at full size">
    <img src="{{ site.baseurl }}/images/kallisto/figure-0-kallisto.jpg" width="675" height="771" loading="lazy" alt="Five-panel overview of kallisto. Three colored transcripts form paths through a transcriptome de Bruijn graph of k-mer nodes. Read k-mers are marked on the graph, dotted arrows skip redundant nodes, and the remaining transcript sets are intersected.">
  </a>
  <figcaption>Figure 1, “Overview of kallisto,” from <a href="https://doi.org/10.1038/nbt.3519">Bray, Pimentel, Melsted, and Pachter (2016)</a>. © 2016 Springer Nature. Select the image to enlarge it.</figcaption>
</figure>

We can understand the contigs concept from the above figure. In panel **b**, each circle is a k-mer and each colored line is a transcript path. Consecutive circles along a nonbranching stretch can be stored as one contig when they carry the same set of transcript colors. In panel **d**, the dotted arrows show kallisto jumping over k-mers whose candidate set would repeat the same information; the labeled nodes are the lookup and checking points. Panel **e** intersects the transcript sets from those informative points to obtain the read's compatibility set.

There is a limit to what the final candidate sets say. They do not retain the order and positions of every match. With very small k, a string can pass the intersection test even when it does not occur as one continuous sequence in a transcript. This is one reason the choice of k matters.

<h2 id="compatibility-classes">3. Compatibility classes: group reads with the same candidates</h2>

After pseudoalignment, many reads have the same candidate set. Grouping them makes the next calculation smaller. Three terms describe the successive stages:

- A **k-mer compatibility set** contains transcripts that contain that k-mer.
- A **read or fragment compatibility set** contains candidates surviving the combined evidence.
- An **equivalence class (EC)** groups observations with the same final candidate set. Its count is the number of observations in that group. These are also called transcript compatibility counts (TCCs).

For example, <span class="k-math" markdown="0">\(\{T_1,T_2\}:90\)</span> means ninety reads could have come from T1 or T2. Their origins are still unresolved. The grouping records one count of ninety, which the abundance model will divide between candidates.

<aside class="k-try" aria-label="Try it" markdown="1">
**Try it:** change the copies of `ACTGACG` from 100 to 200. At k = 3, this adds evidence to the <span class="k-math" markdown="0">\(\{T_1\}\)</span> class. Move the k slider in this experiment and watch the same reads regroup; at k = 7, reads shorter than seven bases become unassigned. Set a row's copies to 0 to remove its evidence, or use **Restore sample** to return to the default counts and k = 3. Press **Use these class counts in EM** to send the assigned counts to the <a href="#em-lab">abundance experiment in section 5</a>.
</aside>

{% include kallisto/class-lab.html %}

### What does grouping preserve?

Consider a compatibility matrix as a table with a row for each read and a column for each transcript. 1 means the transcript remains a candidate; 0 means it does not:

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

In symbols, let <span class="k-math" markdown="0">\(\alpha_t\)</span> be the probability that a sampled fragment comes from <span class="k-math" markdown="0">\(t\)</span>, and <span class="k-math" markdown="0">\(\ell_t\)</span> its effective length, explained next. Let <span class="k-math" markdown="0">\(e\)</span> denote a candidate set and <span class="k-math" markdown="0">\(c_e\)</span> its count. Here <span class="k-math" markdown="0">\(F\)</span> contains the assigned fragments and <span class="k-math" markdown="0">\(E\)</span> contains their equivalence classes. Kallisto's basic likelihood can be written as:

<div class="k-equation">
\[
\begin{aligned}
g_e(\boldsymbol{\alpha})
  &= \sum_{t\in e}\frac{\alpha_t}{\ell_t},
  \quad \alpha_t\geq 0,
  \quad \sum_t\alpha_t=1, \\
\mathcal{L}(\boldsymbol{\alpha})
  &\propto \prod_{f\in F}g_{C(f)}(\boldsymbol{\alpha})
  = \prod_{e\in E}\bigl[g_e(\boldsymbol{\alpha})\bigr]^{c_e}.
\end{aligned}
\]
</div>

For the four fragments above, three belong to the compatibility class <span class="k-math" markdown="0">\(\{T_1,T_2\}\)</span>, and one belongs to <span class="k-math" markdown="0">\(\{T_1\}\)</span>. Therefore, their equivalence-class counts are <span class="k-math" markdown="0">\(c_{\{T_1,T_2\}}=3\)</span> and <span class="k-math" markdown="0">\(c_{\{T_1\}}=1\)</span>. The read-level likelihood <span class="k-math" markdown="0">\(g_{12}g_{12}g_1g_{12}\)</span> can consequently be written as <span class="k-math" markdown="0">\(g_{12}^{3}g_1\)</span>. These **equivalence-class counts** are sufficient statistics for this likelihood: they retain everything needed to calculate it.

<h2 id="effective-length">4. Why effective length appears</h2>

Imagine two transcripts present in equal numbers of RNA molecules, but one offers twice as many possible fragment starts. It can contribute more fragments simply because it is longer. **Fragment share** <span class="k-math" markdown="0">\(\alpha\)</span> therefore differs from the transcript's share of the RNA molecules.

**Effective length** accounts for the available fragment starts. For a transcript of length <span class="k-math" markdown="0">\(L\)</span> and a fixed fragment length <span class="k-math" markdown="0">\(d\leq L\)</span>, a simple effective length is <span class="k-math" markdown="0">\(\ell=L-d+1\)</span>. A 1,000-base transcript with 200-base fragments, for example, has 801 possible starts. Real library models account for a distribution of fragment lengths.

The basic model describes two choices: select transcript <span class="k-math" markdown="0">\(t\)</span> with probability <span class="k-math" markdown="0">\(\alpha_t\)</span>, then select one of its possible fragment locations. The second choice contributes the inverse-length factor. That is why a compatible transcript contributes <span class="k-math" markdown="0">\(\alpha/\ell\)</span> to the likelihood term.

With equal fragment shares but effective lengths 100 and 200, a shared observation has relative weights <span class="k-math" markdown="0">\(\frac{0.5}{100}:\frac{0.5}{200}=2:1\)</span>. You can explore this by changing effective lengths in the <a href="#em-lab">EM controls experiment</a>.

<h2 id="em">5. EM: distribute evidence, then update the estimate</h2>

Suppose unique reads strongly support T1, but only weakly support T2. It would be surprising to divide all reads shared by T1 and T2 equally. Evidence from the whole sample should influence that division.

**Expectation-maximization (EM)** does this in two repeating steps. The **E-step** uses the current abundance estimate to divide ambiguous evidence. The **M-step** adds up those assignments and uses the totals as the next abundance estimate. The origin of each ambiguous fragment is the hidden information being estimated.

For a class <span class="k-math" markdown="0">\(e\)</span>, the fraction allocated to a compatible transcript is called its **responsibility**, written <span class="k-math" markdown="0">\(w_{e,t}\)</span>. It is that transcript's <span class="k-math" markdown="0">\(\alpha/\ell\)</span> weight divided by the total weight of the candidates:

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

The M-step simply divides expected counts by <span class="k-math" markdown="0">\(N\)</span>; <span class="k-math" markdown="0">\(\alpha\)</span> remains a fragment fraction.

### A calculation you can reproduce

Take 100 observations unique to T1, 10 unique to T2, and 90 compatible with both. Use equal effective lengths and equal starting weights. T3 has no supporting class here.

The first E-step splits the ninety shared observations 45/45. Expected counts are <span class="k-math" markdown="0">\((145,55,0)\)</span>, so the M-step yields <span class="k-math" markdown="0">\((0.725,0.275,0)\)</span>. The next E-step allocates <span class="k-math" markdown="0">\(90\times0.725=65.25\)</span> to T1 and 24.75 to T2. Updating gives <span class="k-math" markdown="0">\((0.82625,0.17375,0)\)</span>.

<aside class="k-try" aria-label="Try it" markdown="1">
**Try it:** select **Worked example** and press **Animate E/M from start**. The E-step highlights while the class counts are divided, then the M-step highlights as the transcript shares and trajectory update. The animation shows the early cycles individually and accelerates through later checkpoints until convergence. To reproduce the arithmetic yourself, reset EM, press **E-step: split counts**, check the 45/45 split, and press **M-step: update α**. Repeat once to obtain <span class="k-math" markdown="0">\((0.82625,0.17375,0)\)</span>.
</aside>

{% include kallisto/em-lab.html %}

In this example, the final T1:T2 ratio is set by the unique evidence: <span class="k-math" markdown="0">\(100:10\)</span>. Running EM to convergence gives approximately <span class="k-math" markdown="0">\((0.90909,0.09091,0)\)</span>. The ninety shared fragments add to the estimated counts but cannot distinguish those two transcripts by themselves.

### Why estimated counts and TPM differ

Fractional assignments naturally give noninteger estimated counts. Once EM has fit the fragment shares, the estimated count for a transcript is <span class="k-math" markdown="0">\(N\alpha\)</span>. To account for transcript length, divide these counts by effective length and scale the resulting shares to a total of one million. The result is **TPM**, or transcripts per million:

<div class="k-equation">
\[
\begin{aligned}
r_t &= \frac{n_t}{\ell_t}, \mathrm{TPM}_t = 10^6\frac{r_t}{\sum_j r_j}.
\end{aligned}
\]
</div>

TPM sums to one million when the total is positive. It equals <span class="k-math" markdown="0">\(\alpha\)</span> multiplied by a million only when the effective lengths are equal. Compare the <span class="k-math" markdown="0">\(\alpha\)</span> and TPM columns after changing the lengths: they answer different questions about the same sample. At iteration zero, <span class="k-math" markdown="0">\(N\alpha\)</span> is only an initial guess; after fitting, it is an estimated fragment count rather than a direct count of original RNA molecules.

### Can a stable answer still be ambiguous?

Select **Unidentifiable**. Only <span class="k-math" markdown="0">\(\{T_1,T_2\}\)</span> appears, and the effective lengths are equal. No observation distinguishes the two transcripts. Once <span class="k-math" markdown="0">\(\alpha_3=0\)</span>, the likelihood depends on <span class="k-math" markdown="0">\(\alpha_1+\alpha_2\)</span>, so every split with that sum equal to one fits equally well.

<aside class="k-try" aria-label="Try it" markdown="1">
**Try it:** run with equal starting weights, then click **Start favoring T1** and run again. The T1/T2 splits differ, but the final likelihood is the same. The starting weights selected a split that the observations themselves could not determine.
</aside>

This is a lack of **identifiability**: the evidence does not determine a unique answer. More reads from exactly the same shared class cannot resolve it; distinguishing reads or additional assumptions are needed. A stable optimizer is therefore only one part of interpreting an abundance estimate.

Up to here, we've learned the total original Kallisto algorithm. More detail or advanced method can be found in their manual. In the following, we're going to discuss some interesting research questions I got inspired from it.

<h2 id="neural-network">6. Could a neural network replace EM?</h2>

Neural networks are increasingly used in scientific computing to approximate calculations that would otherwise require an expensive numerical solver. Such a network is often called a **surrogate model**: it is trained on input–output examples from the original calculation and then learns a faster approximation to that mapping. A prominent example is the [Fourier Neural Operator](https://arxiv.org/abs/2010.08895), which learns mappings from the inputs of partial differential equations to their solutions across a family of problems, rather than solving each instance independently from the beginning.

Transcript quantification suggests a related opportunity. After pseudoalignment has reduced the reads to equivalence-class counts, EM repeatedly redistributes those counts and updates transcript abundances until convergence. The 2026 [GPU kallisto study](https://www.biorxiv.org/content/10.64898/2026.03.04.709526v1.full.pdf) makes the remaining cost visible: in its average timing breakdown, GPU mapping takes 722 ms while EM takes 3,148 ms. Accelerating k-mer lookup therefore exposes abundance estimation as a major part of the computation.

There is already research connecting neural networks and EM. [Neural Expectation Maximization](https://papers.neurips.cc/paper/7246-neural-expectation-maximization) constructs a differentiable EM-like procedure in which a neural network learns the statistical model used for perceptual grouping. [UNEM](https://openaccess.thecvf.com/content/CVPR2025/html/Zhou_UNEM_UNrolled_Generalized_EM_for_Transductive_Few-Shot_Learning_CVPR_2025_paper.html) takes another route: it unfolds the iterations of a generalized EM algorithm into network layers and learns iteration-specific parameters for few-shot classification. Neither paper studies RNA-seq, but both show that the repeated structure of EM can be exposed to learning.

This raises a research question for transcript quantification: **can a neural network learn the EM computation that maps equivalence-class evidence to transcript abundance?**

<h2 id="gpu">7. What changes on a GPU?</h2>

Modern GPUs can run thousands of small operations at the same time, and their memory bandwidth has increased along with their computing power. Kallisto contains work that can be separated naturally: different fragments can look up k-mers independently, different candidate sets can be intersected independently, and different equivalence classes can contribute to an EM update in parallel.

### An original kallisto author tries the GPU

In the March 2026 preprint [RNA-seq analysis in seconds using GPUs](https://www.biorxiv.org/content/10.64898/2026.03.04.709526v1.full.pdf), **Páll Melsted**, an author of the original kallisto paper, Elís Mar Guðnýjarson, and Jóhannes Nordal redesign pseudoalignment, equivalence-class intersection, and EM for NVIDIA GPUs. Across 100 Geuvadis RNA-seq samples, they report about a 30× speedup when setup is excluded. A dataset containing 295 million paired-end reads falls from about 40 minutes with 16 CPU threads to 50 seconds on the GPU.

The plot below is their result:

<figure class="k-paper-figure k-paper-benchmark" id="gpu-benchmark">
  <a href="{{ site.baseurl }}/images/kallisto/figure-1-benchmark.jpg" aria-label="Open the original benchmark plot at full size">
    <img src="{{ site.baseurl }}/images/kallisto/figure-1-benchmark.jpg" width="1280" height="1251" loading="lazy" alt="Wall time versus sample read count for 100 Geuvadis samples. CPU kallisto times rise from about 100 to 380 seconds; GPU kallisto times remain below about 20 seconds across the plotted range.">
  </a>
  <figcaption>Figure 1 from <a href="https://doi.org/10.64898/2026.03.04.709526">Melsted, Guðnýjarson, and Nordal (2026)</a></figcaption>
</figure>

The benchmark used an RTX 5090 with 32 GB of GPU memory, a Ryzen 9 9900X, NVMe storage, and BGZF-compressed input. BGZF divides compressed data into independent blocks, allowing several blocks to be decompressed on the GPU at once; ordinary gzip remains serial and is decompressed on the CPU.

### How they implement it: k-mers and EM in parallel

The transcript index is moved into GPU memory. Each k-mer is encoded as a 64-bit integer and looked up in a GPU hash table to obtain an equivalence-class ID. Transcript sets are stored in one flattened array with offsets marking where each set begins. Threads generate and look up k-mers from many reads concurrently, remove repeated and empty candidate sets, intersect the remaining transcript lists, and look up the resulting equivalence class.

The intersections have different sizes, so their memory requirements are not known in advance. The implementation first estimates how much temporary space each read needs, uses a parallel prefix scan to assign each thread a nonoverlapping slice of memory, and then performs the intersections in a second pass. This two-pass design replaces the convenient per-read dynamic allocation that a CPU implementation might use.

<figure class="k-paper-figure" id="gpu-pipeline">
  <a href="{{ site.baseurl }}/images/kallisto/figure-2-pipeline.jpg" aria-label="Open the original GPU pipeline diagram at full size">
    <img src="{{ site.baseurl }}/images/kallisto/figure-2-pipeline.jpg" width="1280" height="621" loading="lazy" alt="Panels A to D connect three transcripts, their colored de Bruijn graph, read k-mer matches, and transcript sets. Panel E follows GPU arrays through k-mer lookup, deduplication, transcript-set intersection, and reverse lookup of the resulting class.">
  </a>
  <figcaption>Figure 2 from <a href="https://doi.org/10.64898/2026.03.04.709526">Melsted et al. (2026)</a>.</figcaption>
</figure>

EM uses a second layout: a transposed index records, for each transcript, every equivalence class containing it. During the E-step, the GPU first calculates the denominator for every class in parallel. It then uses the transposed index to sum class contributions for every transcript in parallel. The M-step normalizes those transcript totals, and the implementation checks convergence every ten iterations. This is the same E-step and M-step from Experiment 3, reorganized so that a single iteration can occupy the GPU.

Detail implementation can be found in [kallisto GPU branch](https://github.com/pachterlab/kallisto/tree/gpu)

### What is the bottleneck now?

The paper reports a mapping rate of 24.1 million read pairs per second, but an end-to-end rate of about 3.6 million pairs per second. In its average timing breakdown, GPU mapping takes 722 ms while EM takes 3,148 ms. I/O and decompression use another 841 ms of GPU time plus 2,370 ms of CPU time. Once k-mer lookup becomes this fast, it is no longer the main constraint.

<figure class="k-paper-figure" id="gpu-runtime">
  <a href="{{ site.baseurl }}/images/kallisto/table-1-runtime.png" aria-label="Open the original runtime table at full size">
    <img src="{{ site.baseurl }}/images/kallisto/table-1-runtime.png" width="2190" height="460" loading="lazy" alt="Table 1: average component runtimes across 100 Geuvadis samples. CPU and GPU times respectively: index setup 376 and 933 ms; I/O and decompression 2370 and 841 ms; GPU mapping 514 and 722 ms; EM has no CPU time listed and 3148 ms GPU time.">
  </a>
  <figcaption>Table 1 from <a href="https://doi.org/10.64898/2026.03.04.709526">Melsted et al. (2026)</a>. The CPU and GPU columns report work within the GPU implementation, rather than two separate implementations.</figcaption>
</figure>

This result shifts the research question from “Can kallisto run on a GPU?” to **Should the saved computation be used to obtain richer evidence, rather than only to reduce runtime?** If pseudoalignment is no longer expensive, selected ambiguous fragments could receive additional alignment or sequence-error scoring before abundance estimation. This possibility leads directly to reconsidering Bowtie2 and RSEM below.

<h2 id="bowtie2-rsem">8. Research direction: revisit Bowtie2 + RSEM on GPUs</h2>

The GPU result suggests a broader question: **if pseudoalignment is now extremely fast, is discarding alignment detail still the best accuracy–runtime tradeoff?** Kallisto keeps candidate-transcript sets, whereas [Bowtie 2](https://pmc.ncbi.nlm.nih.gov/articles/PMC3322381/) preserves base-level alignment evidence and [RSEM](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-12-323) uses that evidence to estimate expression. Modern GPUs may make it practical to retain more information without returning to the runtimes that originally motivated pseudoalignment.

Bowtie2 + RSEM was a strong accuracy baseline in the original kallisto comparison, but that does not establish it as universally more accurate. Later work also shows that [mapping methodology affects abundance accuracy on real data](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-020-02151-8). The research question is therefore whether richer alignment evidence improves transcript estimates enough to justify its remaining computational cost, particularly for rare and highly ambiguous isoforms.

A rough study would have three stages. First, accelerate Bowtie2-compatible alignment while preserving the paired-end relationships, alignment scores, and record formats that [RSEM requires](https://github.com/deweylab/RSEM#using-an-alternative-aligner). Second, parallelize RSEM's abundance estimation without changing its statistical model. Third, compare this pipeline with GPU kallisto on the same references and RNA-seq libraries, measuring end-to-end runtime, memory use, and gene- and transcript-level accuracy using both simulated truth and independent experimental evidence. The result would test whether modern hardware changes which information should be retained for transcript quantification.

<h2 id="conclusion">9. Conclusion: two research directions</h2>

Kallisto became fast by asking only which transcripts are compatible with each fragment, compressing repeated candidate sets into equivalence-class counts, and applying EM to estimate abundance. The experiments in this article expose both sides of that design: the compressed representation makes computation efficient, while ambiguous fragments and iterative abundance estimation remain imperfect.

Two research directions follow from this tension:

- **Learn the abundance calculation.** A neural surrogate could operate on the graph connecting equivalence classes and transcripts, learning several EM-like updates or directly approximating the converged abundance estimate. This direction asks whether the repeated computation can be learned while preserving the likelihood objective and correct behavior when the evidence is ambiguous.
- **Retain richer evidence with GPU computing.** A GPU implementation of Bowtie2 + RSEM could preserve alignment locations, scores, mismatches, and paired-end constraints that pseudoalignment omits. This direction asks whether modern hardware can make a more detailed statistical model competitive in runtime and more informative for rare or ambiguous isoforms.

<h2 id="references">10. References</h2>

<ol class="k-references">
  <li>Bray, N. L., Pimentel, H., Melsted, P., and Pachter, L. (2016). <a href="https://doi.org/10.1038/nbt.3519">Near-optimal probabilistic RNA-seq quantification</a>. <em>Nature Biotechnology</em>, 34, 525–527.</li>
  <li>Pachter Lab. <a href="https://pachterlab.github.io/kallisto/manual">kallisto manual</a>.</li>
  <li>Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., and Anandkumar, A. (2021). <a href="https://arxiv.org/abs/2010.08895">Fourier Neural Operator for Parametric Partial Differential Equations</a>. <em>International Conference on Learning Representations</em>.</li>
  <li>Greff, K., van Steenkiste, S., and Schmidhuber, J. (2017). <a href="https://papers.neurips.cc/paper/7246-neural-expectation-maximization">Neural Expectation Maximization</a>. <em>Advances in Neural Information Processing Systems</em>, 30.</li>
  <li>Zhou, L., Shakeri, F., Sadraoui, A., Kaaniche, M., Pesquet, J.-C., and Ben Ayed, I. (2025). <a href="https://openaccess.thecvf.com/content/CVPR2025/html/Zhou_UNEM_UNrolled_Generalized_EM_for_Transductive_Few-Shot_Learning_CVPR_2025_paper.html">UNEM: UNrolled Generalized EM for Transductive Few-Shot Learning</a>. <em>Proceedings of CVPR</em>, 9665–9675.</li>
  <li>Melsted, P., Guðnýjarson, E. M., and Nordal, J. (2026). <a href="https://doi.org/10.64898/2026.03.04.709526">RNA-seq analysis in seconds using GPUs</a>. <em>bioRxiv</em>, version 1.</li>
  <li>Pachter Lab. <a href="https://github.com/pachterlab/kallisto/tree/gpu">GPU branch of kallisto</a>. Source code accompanying Melsted et al. (2026).</li>
  <li>Langmead, B., and Salzberg, S. L. (2012). <a href="https://doi.org/10.1038/nmeth.1923">Fast gapped-read alignment with Bowtie 2</a>. <em>Nature Methods</em>, 9, 357–359.</li>
  <li>Li, B., and Dewey, C. N. (2011). <a href="https://doi.org/10.1186/1471-2105-12-323">RSEM: accurate transcript quantification from RNA-Seq data with or without a reference genome</a>. <em>BMC Bioinformatics</em>, 12, 323. See also the <a href="https://github.com/deweylab/RSEM#using-an-alternative-aligner">RSEM alignment requirements</a>.</li>
  <li>Srivastava, A., Malik, L., Sarkar, H., Zakeri, M., Almodaresi, F., Soneson, C., Love, M. I., Kingsford, C., and Patro, R. (2020). <a href="https://doi.org/10.1186/s13059-020-02151-8">Alignment and mapping methodology influence transcript abundance estimation</a>. <em>Genome Biology</em>, 21, 239.</li>
</ol>

<script defer src="{{ site.baseurl }}/assets/kallisto/vendor/katex-0.18.7/katex.min.js"></script>
<script defer src="{{ site.baseurl }}/assets/kallisto/math.js"></script>
<script defer src="{{ site.baseurl }}/assets/kallisto/model.js"></script>
<script defer src="{{ site.baseurl }}/assets/kallisto/playground.js"></script>
