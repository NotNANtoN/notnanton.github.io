---
title: Using video models to learn robot actions
description: Offline experiments with Cosmos and LTX video features, task video fine-tuning, and representation distillation for an SO-101 robot.
date: 2026-08-29
tags:
  - robotics
  - video-models
  - world-models
  - imitation-learning
authors:
  - name: Anton Wiehe
    url: https://notnanton.github.io
    affiliation: PHAROS Labs / AdaLab
featured: true
thumbnail: /assets/img/video-vam/lora-ep19-contact-sheet.png
giscus: true
---

Can pretrained video representations help a robot learn actions from a few demonstrations? Video models are trained to predict how scenes change; I wanted to test whether those features also help with action learning. I tried NVIDIA Cosmos-Predict2 2B and Lightricks LTX-2.5 as feature extractors for an SO-101 learning to take a cube out of a box, using one RTX 4090.

I evaluated predicted actions on held-out recordings; I have not measured task success on the robot yet. The implementation builds on [LeRobot](https://github.com/huggingface/lerobot) and [mimic-video](https://github.com/mimic-video/mimic-video). Video generation offers a useful visual check, but the policy has a different job: predict actions from internal features.

<figure class="post-figure">
  <video controls loop muted playsinline preload="metadata" width="1920" height="544" poster="/assets/img/video-vam/cosmos-ep19-three-way.jpg">
    <source src="/assets/video/video-vam/cosmos-ep19-three-way.mp4" type="video/mp4">
  </video>
  <figcaption>Left: base Cosmos. Center: video-LoRA Cosmos. Right: recorded ground truth. Both models receive the same five observed RGB frames and predict 5.6 seconds of future video. Episode 19 is a training example, not held-out evaluation.</figcaption>
</figure>

These clips use repeated denoising. The action policy does not generate videos: it reads features from a single backbone forward pass.

## Data and measurement

The benchmark uses [hubnemo/cube_out_of_box_dataset, pinned to revision 243370c3c08bcbd860133c4a0d658ea7c1d2e77e](https://huggingface.co/datasets/hubnemo/cube_out_of_box_dataset/tree/243370c3c08bcbd860133c4a0d658ea7c1d2e77e): the v1 dataset, not v2. There are 40 episodes: 32 training episodes (0–31) and eight validation episodes (32–39). Evaluation uses 88 sampled observation windows, a 30-step action horizon, and single-camera observations at 10 Hz and 480×640 resolution.

The metric is masked global aggregate RMSE over valid action targets; lower is better. It combines five angular joints with a gripper channel scaled 0–100, so the aggregate is **not an angle in degrees**. Some runs used different preprocessing or evaluation settings, so the table below records the experiments rather than a definitive ranking.

## From video features to actions

The Cosmos path is:

1. A causal video VAE encodes five observed RGB frames into two latent frames. Here, T counts latent frames, not RGB frames.
2. The T=16 path adds 14 noisy future latent slots, not actual future observations. One DiT forward stops at layer 20. Its 19,200 raw tokens become 4,800 tokens through factor-two spatial pooling across all 16 slots.
3. The observed-only T=2 path omits those future slots and retains 2,400 unpooled tokens. It changes the computation as well as the number of tokens.
4. A small pretrained SmolExpert action head reads the visual features through prefix key/value attention, alongside current joint state, and uses flow matching to predict the 30-step action chunk.

I freeze the backbone and cache features before training the action head. That makes head experiments cheaper to repeat, but does not remove feature extraction during live operation. LTX uses a separate extractor with a block-34 feature tap.

## Recorded results

| Configuration | Aggregate RMSE |
|---|---:|
| Repeat current joint state | 18.86 |
| SmolVLA, train-only normalization | 14.93 |
| Frozen Cosmos + World2Action | 14.51 |
| Frozen Cosmos + SmolExpert | 13.65 |
| Video-LoRA Cosmos, T=16 + SmolExpert | 13.06 |
| Video-LoRA Cosmos, T=2 + SmolExpert | 13.74 |
| Frozen LTX-2.5 + SmolExpert | 13.84 |

Two changes need separating. On identical frozen Cosmos features, replacing the from-scratch World2Action decoder with pretrained SmolExpert changed the recorded result from 14.51 to 13.65. SmolExpert reached its best checkpoint in about 2.5 hours of head training.

Task video adaptation was a separate experiment: train a video-prediction LoRA, freeze it, then train the action head. That run reached 13.06. However, a frozen Cosmos alternative using causal-prefix VAE encoding recorded 13.81 rather than the older 13.65. The preprocessing and evaluation history prevents treating the difference as a clean LoRA-only effect.

One joint video/action training configuration performed poorly. That is evidence about that configuration, not proof that co-training is inherently wrong. Likewise, these results do not establish a general advantage for smaller models or better sample efficiency.

## What got faster

Cosmos feature extraction measured 1,163 ms for T=16 and 204 ms for T=2: a 5.7× speedup on these paths. The associated action error increased by 0.68, from 13.06 to 13.74. Because the policy also changes pooling and feature context, this is not a pure frame-count ablation.

The T=2 head reached its best checkpoint after about 46 minutes, excluding feature extraction and video-LoRA pretraining. Extraction latency is not end-to-end reaction time.

For CPU-streamed LTX, reducing the latent sequence from eight frames to two barely changed extraction: 1,243 versus 1,228 ms. GPU-resident INT4 reached 780 ms, but its downstream policy quality has not been validated.

## Representation distillation

Video-generation loss is not the only way to adapt the observed-only representation. I also tried matching features directly:

- A frozen T=16 teacher receives the same causal five-frame RGB input as the T=2 student. After the full-context forward, its layer-20 features from the **first two conditioning slots** become the targets; these are not future-video targets.
- The student trains LoRA adapters in blocks 0–19 to match those cached teacher features with squared-error and cosine-similarity losses.
- I then freeze the student, cache its features, and train the action head separately.

The initial runs improved action prediction. I want matched re-evaluations before quoting a score. Early experiments were muddied by a four-frame offset between teacher and student input windows. We also changed the training targets and auxiliary readout head, so those runs cannot tell us which change helped most.

## Limits and next tests

This is one task and a small dataset. Posture/session statistics shift around episodes 20–24; an operator change or recalibration is a hypothesis, not an established cause. Validation lies after that shift. Sharing the confound does not guarantee that model rankings survive a cleaner split.

Next come matched evaluation and real-robot tests, including responsive replanning. Real-time chunking overlaps inference with execution and guides replacement chunks, but smoother chunk boundaries are not task accuracy. Guided end-to-end performance and robot success rates remain unverified.
