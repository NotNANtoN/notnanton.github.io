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

I evaluated predicted actions on held-out recordings; I have not measured task success on the robot yet. The implementation builds on [LeRobot](https://github.com/huggingface/lerobot) and [mimic-video](https://github.com/mimic-video/mimic-video). Upstream mimic-video couples a video world model with a from-scratch flow-matching DiT action decoder (referred to as World2Action). Video generation offers a useful visual check, but the policy has a different job: predict actions from internal features.

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

Here, T counts latent frames, not RGB frames. The two Cosmos paths use the same video-LoRA backbone, but different input lengths and policy features.

<figure class="post-figure">
  <iframe src="/assets/interactive/cosmos-feature-paths.html" title="Interactive Cosmos feature paths: reference T=16 and proposed T=2" width="1200" height="800" loading="lazy" style="display:block;width:100%;height:800px;border:0;background:#fff"></iframe>
  <figcaption>Switch paths to follow the layer-20 readout into SmolExpert. These undistilled paths share the video-LoRA weights; T=2 omits 14 latent noise slots. Timings cover feature extraction only, and pooling also changes. <a href="/assets/interactive/cosmos-feature-paths.html" target="_blank" rel="noopener">Open the interactive figure full size.</a></figcaption>
</figure>

To map video representations to robot trajectories, I benchmarked two action head architectures:
1. **World2Action**: The native mimic-video action head, a flow-matching Diffusion Transformer trained from scratch to cross-attend to the video tokens.
2. **SmolExpert**: An action head initialized from SmolVLA's pretrained action expert. It reads visual features through prefix key/value attention, combines them with current joint state, and predicts a 30-step action chunk via flow matching.

I freeze the backbone and cache features before training the action head. That makes head experiments cheaper to repeat, but does not remove feature extraction during live operation. LTX uses a separate extractor with a block-34 feature tap.

## Recorded results

| Configuration | Aggregate RMSE |
|---|---:|
| Repeat current joint state | 18.86 |
| SmolVLA, train-only normalization | 14.93 |
| Frozen Cosmos + World2Action (mimic-video from-scratch DiT) | 14.51 |
| Frozen Cosmos + SmolExpert (SmolVLA pretrained head) | 13.65 |
| Video-LoRA Cosmos, T=16 + SmolExpert | 13.06 |
| Video-LoRA Cosmos, T=2 + SmolExpert | 13.74 |
| Frozen LTX-2.5 + SmolExpert | 13.84 |

Two changes need separating. First, the action decoder: on identical frozen Cosmos features, replacing the native from-scratch mimic-video World2Action decoder with the pretrained SmolExpert action head improved the recorded score from 14.51 to 13.65. Reusing pretrained action weights from SmolVLA delivered superior performance over the from-scratch mimic-video DiT and converged quickly, reaching its best checkpoint in about 2.5 hours of head training.

Task video adaptation was a separate experiment: train a video-prediction LoRA, freeze it, then train the action head. That run reached 13.06. However, a frozen Cosmos alternative using causal-prefix VAE encoding recorded 13.81 rather than the older 13.65. The preprocessing and evaluation history prevents treating the difference as a clean LoRA-only effect.

One joint video/action training configuration performed poorly. That is evidence about that configuration, not proof that co-training is inherently wrong. Likewise, these results do not establish a general advantage for smaller models or better sample efficiency.

## What got faster

Cosmos feature extraction measured 1,163 ms for T=16 and 204 ms for T=2: a 5.7× speedup on these paths. The associated action error increased by 0.68, from 13.06 to 13.74. Because the policy also changes pooling and feature context, this is not a pure frame-count ablation.

The T=2 head reached its own best checkpoint after about 46 minutes of head training on cached features, excluding feature extraction and video-LoRA pretraining. Extraction latency is not end-to-end reaction time.

<figure class="post-figure" data-zoomable>
  <img src="/assets/img/video-vam/cosmos-t16-t2-training.svg" alt="Two stacked panels comparing validation action RMSE for T=16 and T=2 against optimizer steps and elapsed head-training loop time." loading="lazy">
  <figcaption>Two archived single-seed W&B runs: T=16 with pool2 (blue) and T=2 unpooled (orange), using the same video-LoRA backbone, batch size 8, and 88 fixed held-out observation windows. Panels show validation action RMSE against optimizer steps and elapsed head-training loop time on cached features; time excludes video-LoRA training and cache creation.</figcaption>
</figure>

The T=2 run processed head-training steps faster, but did not reach every error threshold sooner. Cheaper feature extraction should also reduce cache-building time; comparable total cache-build timings were not recoverable for these runs.

For CPU-streamed LTX, reducing the latent sequence from eight frames to two barely changed extraction: 1,243 versus 1,228 ms. GPU-resident INT4 reached 780 ms, but its downstream policy quality has not been validated.

## Representation distillation

Distillation tries to keep the short inference path while training its features to resemble those from the longer path. Instead of using video-generation loss alone, I also tried matching features directly:

- A frozen T=16 teacher receives the same causal five-frame RGB input as the T=2 student. After the full-context forward, its layer-20 features from the **first two conditioning slots** become the targets; these are not future-video targets.
- The student trains LoRA adapters in blocks 0–19 to match those cached teacher features with squared-error and cosine-similarity losses.
- I then freeze the student, cache its features, and train the action head separately.

The initial runs improved action prediction. I want matched re-evaluations before quoting a score. Early experiments were muddied by a four-frame offset between teacher and student input windows. We also changed the training targets and auxiliary readout head, so those runs cannot tell us which change helped most.

## Limits and next tests

This is one task and a small dataset. Posture/session statistics shift around episodes 20–24; an operator change or recalibration is a hypothesis, not an established cause. Validation lies after that shift. Sharing the confound does not guarantee that model rankings survive a cleaner split.

Next come matched evaluation and real-robot tests, including responsive replanning. Real-time chunking overlaps inference with execution and guides replacement chunks, but smoother chunk boundaries are not task accuracy. Guided end-to-end performance and robot success rates remain unverified.
