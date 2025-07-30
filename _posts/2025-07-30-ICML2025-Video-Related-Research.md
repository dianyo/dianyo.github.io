---
layout: post
title: ICML2025 Video-Related Research
---

In this post, I'll quickly summarize the video generation related research that I found interesting in ICML2025.

## XAttention: Block Sparse Attention with Antidiagonal Scoring

### Problem
Existing block-sparse methods have struggled to deliver on their full potential of orginal models, often grappling with a trade-off between accuracy and efficiency, where the efficiency is also limited by importance score searching. **Can we design a block-sparse attention mechanism that dramatically accelerates longcontext Transformers without compromising accuracy, truly unlocking their potential for real-world applications?**

<div style="text-align: center; width: 60%; margin: 0 auto;">
<img src="https://dianyo.github.io//images/ICML2025_videogen/XAttn_search_compare.png" alt="XAttn_search_compare">
</div>

### Proposed Method
The authors propose a new block-sparse attention mechanism, XAttention, that leverages antidiagonal scoring to achieve high accuracy while maintaining efficiency by their empirical observations though they didn't mention a lot of how they find this observation. Other than the block selection by a threshold, the authors also propose a dynamic threshold prediction method using dynamic programming to set the threshold for each block but which is not mandatory.

<div style="text-align: center; width: 100%; margin: 0 auto;">
<img src="https://dianyo.github.io//images/ICML2025_videogen/XAttn_fig.png" alt="XAttention" style="width: 48%; display: inline-block;">
<img src="https://dianyo.github.io//images/ICML2025_videogen/XAttn_algo.png" alt="XAttention algorithm" style="width: 48%; display: inline-block;">
</div>

### Experiments & Results

As the XAttention is focus on prefill stage of the Transformer, it can be applied to any Transformer-based models. We only show the results on video generation task using HunyuanVideo here, but the authors also show the results on other tasks like NLP using Llama-3.18B-Instruc and video understanding using Qwen2-VL-7B-Instruct. Authors choose 946 GPT-augmented text prompts from VBench for video generation task, and use full attention as the baseline as the above mentioned methods are all applied to casual attention.

Authors found that applying XAttention from the very beginning of the denoising process in the HunyuanVideo model led to slight layout shifts, that they decided to introduce a 5-step "warmup" stage as research shows early denoising steps are critical for determining content layout. The reults shows more than 50% sparsity can be achieved after applying XAttention, however they didn't provide the speedup numbers directly.

<div style="text-align: center; width: 60%; margin: 0 auto;">
<img src="https://dianyo.github.io//images/ICML2025_videogen/XAttn_video_warmup.png" alt="XAttn_warmup">
</div>

<div style="text-align: center; width: 60%; margin: 0 auto;">
<img src="https://dianyo.github.io//images/ICML2025_videogen/XAttn_video_table.png" alt="XAttn_results">
</div>

## ConceptAttention: Diffusion Transformers Learn Highly Interpretable Features

### Problem
The understanding of the internal mechanisms of diffusion models is limited, and the decision-making process of diffusion models is not interpretable. The rapid advancement and enhanced capabilities of DiT-based models highlight the critical importance of methods that improve their interpretability, transparency, and safety.

### Proposed Method
ConceptAttention utilize the multi-modal attention layers (MMATTN) in DiT to generate high quality saliency maps that depict the location of the input textual concepts in generated images. The authors create a set of contextualized concept embeddings for textual concepts and use them alongside the original input image and text, the concept input will do cross-attention with the image and self-attention with itself cause the authors found that performing both instead of just cross-attention improves the donwstream tasks performance. The concept branch won't affect the orignal generation process. The high level idea is shown in the following figure

<div style="text-align: center; width: 60%; margin: 0 auto;">
<img src="https://dianyo.github.io//images/ICML2025_videogen/ConceptAttn_fig.png" alt="ConceptAttention">
</div>

The detail algorithm is composed by the following fomulas, where subscripts $x$ denotes image, $p$ denotes text input prompt, $c$ denotes concept input.

<div style="text-align: center; width: 60%; margin: 0 auto;">
<img src="https://dianyo.github.io//images/ICML2025_videogen/ConceptAttn_formula4.png" alt="ConceptAttention formula 4">
<img src="https://dianyo.github.io//images/ICML2025_videogen/ConceptAttn_formula5.png" alt="ConceptAttention formula 5">
<img src="https://dianyo.github.io//images/ICML2025_videogen/ConceptAttn_formula6.png" alt="ConceptAttention formula 6">
<img src="https://dianyo.github.io//images/ICML2025_videogen/ConceptAttn_formula7.png" alt="ConceptAttention formula 7">
<img src="https://dianyo.github.io//images/ICML2025_videogen/ConceptAttn_formula8.png" alt="ConceptAttention formula 8">
<img src="https://dianyo.github.io//images/ICML2025_videogen/ConceptAttn_formula9_10.png" alt="ConceptAttention formula 9 & 10">
<img src="https://dianyo.github.io//images/ICML2025_videogen/ConceptAttn_formula11.png" alt="ConceptAttention formula 11">
<img src="https://dianyo.github.io//images/ICML2025_videogen/ConceptAttn_formula_sum.png" alt="ConceptAttention formula final">
</div>

### Experiments & Results
Most of the experiments are conducted on image generation task, but author shows in one case that ConceptAttention can also be applied to video generation task on CogVideoX, where they further average over the frame dimension to get the final saliency map.

<div style="text-align: center; width: 60%; margin: 0 auto;">
<img src="https://dianyo.github.io//images/ICML2025_videogen/ConceptAttn_video_result.png" alt="ConceptAttention video results">
</div>

The authors also provide a inspiring ablation study on how different the output concept various between diffusion steps and DiT layers. The layers is more intuitive that the deeper layers have more refined representation that better transfer to the segmentation task. However, for the diffusion steps, it's surprising that although the later timesteps is less noisy, the concpet output is the best at the middle of the diffusion process.

<div style="text-align: center; width: 60%; margin: 0 auto;">
<img src="https://dianyo.github.io//images/ICML2025_videogen/ConceptAttn_ablation.png" alt="ConceptAttention ablation">
</div>
