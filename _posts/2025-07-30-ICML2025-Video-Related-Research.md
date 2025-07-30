---
layout: post
title: ICML2025 Video-Related Research
---

In this post, I'll quickly summarize the video generation related research that I found interesting in ICML2025.

## XAttention: Block Sparse Attention with Antidiagonal Scoring

### Problem
Existing block-sparse methods have struggled to deliver on their full potential of orginal models, often grappling with a trade-off between accuracy and efficiency, where the efficiency is also limited by importance score searching. **Can we design a block-sparse attention mechanism that dramatically accelerates longcontext Transformers without compromising accuracy, truly unlocking their potential for real-world applications?**

![XAttn_search_compare](/images/ICML2025_videogen/XAttn_search_compare.png)


### Proposed Method
The authors propose a new block-sparse attention mechanism, XAttention, that leverages antidiagonal scoring to achieve high accuracy while maintaining efficiency by their empirical observations though they didn't mention a lot of how they find this observation. Other than the block selection by a threshold, the authors also propose a dynamic threshold prediction method using dynamic programming to set the threshold for each block but which is not mandatory.
| ![XAttention](/images/ICML2025_videogen/XAttn_fig.png)        | ![XAttention algorithm](/images/ICML2025_videogen/XAttn_algo.png) |
|:-------------------------------------------------------------:|:-------------------------------------------:|


### Experiments & Results

As the XAttention is focus on prefill stage of the Transformer, it can be applied to any Transformer-based models. We only show the results on video generation task using HunyuanVideo here, but the authors also show the results on other tasks like NLP using Llama-3.18B-Instruc and video understanding using Qwen2-VL-7B-Instruct. Authors choose 946 GPT-augmented text prompts from VBench for video generation task, and use full attention as the baseline as the above mentioned methods are all applied to casual attention.

Authors found that applying XAttention from the very beginning of the denoising process in the HunyuanVideo model led to slight layout shifts, that they decided to introduce a 5-step "warmup" stage as research shows early denoising steps are critical for determining content layout. The reults shows more than 50% sparsity can be achieved after applying XAttention, however they didn't provide the speedup numbers directly.

![XAttn_warmup](/images/ICML25_videogen/XAttn_video_warmup.png)
![XAttn_results](/images/ICML25_videogen/XAttn_video_table.png)





