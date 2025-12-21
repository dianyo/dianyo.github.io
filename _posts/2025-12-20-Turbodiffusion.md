---
layout: post
title: How TurboDiffusion accerelate video diffusion models by 100-200x? A detail methods review including SageAttention, Sparse-Linear Attention, rCM and INT8 block-wise quantization.
---

TurboDiffusion is a new video diffusion generation framework annouced about a week ago (2025-12-13). It claims to accelerate video diffusion models by 100-200x, using Wan-2.1 as the baseline. In this post, I'll review the main techniques used in TurboDiffusion and how they achieve the acceleration.

First, TurboDiffusion uses SageAttention and Sparse-Linear Attention (SLA) to speed up attention computation. SageAttention is a family of low-bit quantization methods for attention, with the first-generation method accepted at ICLR 2025. This line of work was later extended by SageAttention2, SageAttention2++, and SageAttention3 (NeurIPS 2025 Spotlight). In TurboDiffusion, SageAttention2++ is used specifically. We’ll walk through the evolution of SageAttention in the first section. SLA is another attention acceleration technique adopted by TurboDiffusion. It employs a trainable attention mechanism that fuses sparse and linear attention to further accelerate diffusion inference. We’ll discuss SLA in more detail in the second section.

Interestingly, both SageAttention and SLA are proposed by the same author, and TurboDiffusion can be viewed as a natural extension that combines these prior works as modular building blocks.

Second, TurboDiffusion adopts rCM, proposed by Nvidia, to accelerate diffusion process by reducing the number of sampling steps. The full name of rCM is *score-regularized continuous-time consistency model*, which belongs to the consistency model (CM) family. Consistency model is a distillation based method, and we'll review the consistency models and rCM in our third section.

Finally, TurboDiffusion uses INT8 (W8A8) block-wise quantization to speed up the linear layers computation. We'll discuss the quantization methods in our fourth section.

After reviewing each individual technique, we’ll look at how they are combined in TurboDiffusion and how their effects stack together to achieve the reported acceleration. And ofc, we'll also review the performance results shared by the authors in this section. If you’re already familiar with the individual methods, feel free to skip directly to this part.

In the end, I’ll share my thoughts on TurboDiffusion, along with some personal takeaways and potential future directions from my perspective.

Without further ado, let’s get started.
