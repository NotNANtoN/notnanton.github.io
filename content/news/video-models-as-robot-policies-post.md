---
title: New post — video diffusion models as robot policies
date: 2026-08-29
inline: false
---

Published a deep-dive on turning open video diffusion models (Cosmos-Predict2 2B, LTX-2.5) into robot policy backbones on a single RTX 4090 — [read it here](/blog/2026/video-models-as-robot-policies/). A LoRA fine-tuned purely on video prediction of the robot's own data gave the best action policy, and dropping the future frames from the forward pass made feature extraction 5.7x faster. Below: the fine-tuned model predicting a held-out episode (left prediction, right ground truth).

<figure class="post-figure">
  <video autoplay loop muted playsinline preload="metadata">
    <source src="/assets/video/video-vam/lora-ep19-side-by-side.mp4" type="video/mp4">
  </video>
</figure>
