---
title: "Video Diffusion Models as Robot Policy Backbones: Cosmos, LTX, and the Race Below One Second"
description: Two weeks of turning open video diffusion models into robot policies — feature extraction, LoRA fine-tuning, a 5.7x latency win, and the dataset confound that humbled every number.
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

## Overview

Video diffusion models learn dynamics: how objects move, how hands interact with them, how scenes evolve. That makes them a tempting foundation for robot policies — a model that can predict video of a manipulation task plausibly *understands* the task in ways a static image encoder does not. Black Forest Labs and mimic robotics demonstrated this direction with [FLUX-mimic](https://bfl.ai/blog/flux-3-mimic), reporting sub-80 ms world-representation latency on an RTX 5090 — but the recipe (quantization, backbone surgery, caching) is not public.

This post documents my attempt to build that recipe in the open, on one RTX 4090, using two open video models — NVIDIA's **Cosmos-Predict2 2B** and Lightricks' **LTX-2.5** — as frozen and fine-tuned feature backbones for a small flow-matching action head, evaluated on a real SO-101 dataset. It covers roughly two weeks of work (2026-08-17 to 2026-08-29), and it is written the way the work happened: results first, then the journey with its dead ends, then the latency engineering, and finally the confound that qualifies everything.

There are no robot deployment videos yet — I'm not happy enough with the policies to film them. What I do have is the thing video models are uniquely able to show: **what the policy backbone believes will happen next.**

<figure class="post-figure">
  <video autoplay loop muted controls playsinline preload="metadata">
    <source src="/assets/video/video-vam/lora-ep19-side-by-side.mp4" type="video/mp4">
  </video>
  <figcaption>LoRA-fine-tuned Cosmos-Predict2 2B predicting 5.6 s of the manipulation task from 5 observed frames (left: prediction, right: ground truth). This same forward pass, stopped at layer 20, is what the action policy reads.</figcaption>
</figure>

This work builds on [my earlier robot RL pipeline](/blog/2026/robot-rl-experiments/) (same SO-101 arm), on [LeRobot](https://github.com/huggingface/lerobot), and on the public [mimic-video](https://github.com/mimic-video/mimic-video) reference implementation.

## Results at a Glance

Task: take a cube out of a box. Dataset: 40 teleoperated episodes (~6,500 frames at 10 Hz, 480×640, single camera), train on episodes 0–31, validate on 32–39. Metric: masked RMSE over 30-step action chunks on held-out anchors, in degrees (five arm joints; the gripper channel is 0–100, so the aggregate is a convention, not a physical angle). All numbers from one RTX 4090.

| Policy backbone | Val RMSE ° | Steps to best | Time to best | Feature extraction / window |
|---|---:|---:|---:|---:|
| mean action (baseline) | 30.15 | — | — | — |
| repeat last joint state (baseline) | 18.86 | — | — | — |
| SmolVLA (converged; batch-64 rerun matched at 14.82) | 14.83 | 29.2k | ~1 h | 265 ms (full policy) |
| Frozen Cosmos 2B, pooled + World2Action head (499M, from scratch) | 14.51 | 19.5k | overnight | 1,163 ms |
| Frozen Cosmos 2B, pooled + SmolExpert head (pretrained, ~100M) | 13.65 | 22k | ~3.5 h | 1,163 ms |
| LTX-2.5, unpooled | 13.84 | 45k | 3.0 h | ~2,000 ms |
| Cosmos 2B + video LoRA, T=2 observed-only | 13.74 | 27k | **46 min** | **204 ms** |
| Cosmos 2B + video LoRA, full 16-frame | **13.06** | 38k | 1.9 h | 1,163 ms |

Three headline findings:

1. **Fine-tune the video model on video prediction, then freeze it.** A LoRA trained purely on next-frame video prediction of the robot's own data improves downstream action RMSE from 14.51° to 13.06° — while *joint* video+action co-training collapsed to 24.37°.
2. **You can throw away 87% of the tokens.** Running the video model on only the 2 observed latent frames instead of the full 16-frame (observed + noisy future) sequence gives a **5.7× extraction speedup** (204 ms vs 1,163 ms) at a cost of 0.68°.
3. **The action expert is nearly free.** With backbone features cached, the 115M-parameter action head reaches its best score in under an hour on the T=2 features.
4. **Bigger was not better — for us.** The 22B LTX-2.5 produced the best *frozen* features (13.84°) but never won overall: it is too large to fine-tune on this hardware, and its CPU-weight-streaming inference (~2 s per window) is the least deployable. The LoRA-tuned 2B Cosmos beats it on both quality and latency.

<figure class="post-figure" data-zoomable>
  <img src="/assets/img/video-vam/vam-smolexpert-training-efficiency.png" alt="Validation RMSE versus wall-clock training hours for all policy arms">
  <figcaption>Validation RMSE vs. wall-clock expert training time. The video-feature policies train fast because the expensive part — the backbone forward — is precomputed into a feature cache.</figcaption>
</figure>

## The Architecture

The design follows the publicly documented part of the FLUX-mimic / mimic-video recipe: **the video model never generates video at inference time.** It runs a single forward pass and the action head reads an internal representation.

<figure class="post-figure" data-zoomable>
  <img src="/assets/img/video-vam/architecture.svg" alt="Pipeline: 5 RGB frames through the video VAE and the Cosmos DiT (LoRA on blocks 0-19, layer-20 tap, blocks 20-27 skipped) into 2,400 tokens consumed as prefix K/V by the action expert; a dashed world-expert branch reads the same interface at training time">
  <figcaption>The full pipeline. One trunk forward per action chunk; the layer-20 tap feeds the action expert as prefix K/V. The dashed world-expert branch consumes the same interface during training and for video evaluation only.</figcaption>
</figure>

The action expert consumes backbone tokens as **prefix key/value pairs**: on even layers the action tokens self-attend with the backbone K/V concatenated in; on odd layers they cross-attend to the backbone tokens only. Actions are generated by flow matching from noise, conditioned on the current joint state. The backbone is frozen during action training — features are extracted once into a hash-validated cache, which is what makes a full policy training run cost minutes-to-hours instead of days.

Two backbones went through this pipeline:

- **Cosmos-Predict2 2B** (video-to-world variant): 28 DiT blocks, width 2048. The 16-frame forward conditions on 2 observed latent frames and fills 14 future slots with high-sigma noise — a *predictive* representation, not just an encoding.
- **LTX-2.5** (~22B DiT): a much larger, FP8-quantized model that must stream weights from CPU on a 24 GB card. Strong features (13.84° unpooled), but a very different latency profile — more below.

## The Journey

The recipe, end to end — and where the open experiment sits in it:

<div class="vam-recipe">
  <p class="vr-hint">Hover (or tap) a step for the details.</p>
  <div class="vr-step" tabindex="0" style="--vr-accent:#64748b">
    <span class="vr-num">1</span>
    <div class="vr-body">
      <p class="vr-title">Take a pretrained video model</p>
      <div class="vr-detail"><p>Cosmos-Predict2 2B or LTX-2.5 &mdash; an internet-scale video prior over objects, contact, and physics. Everything downstream reads this model&rsquo;s internal representation.</p></div>
    </div>
    <svg class="vr-illus" viewBox="0 0 150 62" aria-hidden="true">
      <g stroke="#64748b" fill="none" stroke-width="1.6">
        <rect x="14" y="8" width="34" height="26" rx="3" fill="#e2e8f0" stroke="#94a3b8"/>
        <rect x="9" y="13" width="34" height="26" rx="3" fill="#f1f5f9" stroke="#94a3b8"/>
        <rect x="4" y="18" width="34" height="26" rx="3" fill="#ffffff"/>
        <path d="M17 25 L27 31 L17 37 z" fill="#64748b" stroke="none"/>
        <line x1="44" y1="31" x2="60" y2="31"/>
        <path d="M60 27 L66 31 L60 35 z" fill="#64748b" stroke="none"/>
      </g>
      <g fill="#cbd5e1">
        <rect x="72" y="13" width="9" height="9" rx="1.5"/><rect x="84" y="13" width="9" height="9" rx="1.5"/><rect x="96" y="13" width="9" height="9" rx="1.5"/><rect x="108" y="13" width="9" height="9" rx="1.5"/><rect x="120" y="13" width="9" height="9" rx="1.5"/>
        <rect x="72" y="25" width="9" height="9" rx="1.5"/><rect x="84" y="25" width="9" height="9" rx="1.5"/><rect x="96" y="25" width="9" height="9" rx="1.5"/><rect x="108" y="25" width="9" height="9" rx="1.5"/><rect x="120" y="25" width="9" height="9" rx="1.5"/>
        <rect x="72" y="37" width="9" height="9" rx="1.5"/><rect x="84" y="37" width="9" height="9" rx="1.5"/><rect x="96" y="37" width="9" height="9" rx="1.5"/><rect x="108" y="37" width="9" height="9" rx="1.5"/><rect x="120" y="37" width="9" height="9" rx="1.5"/>
      </g>
    </svg>
    <span class="vr-chev" aria-hidden="true"></span>
  </div>
  <div class="vr-arrow" aria-hidden="true"></div>
  <div class="vr-step vr-optional" tabindex="0" style="--vr-accent:#94a3b8">
    <span class="vr-num">2</span>
    <div class="vr-body">
      <p class="vr-title">Robot-domain fine-tune <span class="vr-tag">optional &mdash; we skipped it</span></p>
      <div class="vr-detail"><p>Video prediction on large cross-robot corpora such as Bridge or Open X-Embodiment before touching your own robot. We went straight to task data; this is the obvious untested lever.</p></div>
    </div>
    <svg class="vr-illus" viewBox="0 0 150 62" aria-hidden="true">
      <g stroke="#94a3b8" fill="none" stroke-width="1.6" stroke-linecap="round">
        <path d="M6 20 L14 14 L22 18"/><circle cx="6" cy="20" r="2.4" fill="#94a3b8"/><circle cx="14" cy="14" r="2" fill="#94a3b8"/><circle cx="22" cy="18" r="2" fill="#94a3b8"/>
        <path d="M6 38 L15 33 L23 37"/><circle cx="6" cy="38" r="2.4" fill="#94a3b8"/><circle cx="15" cy="33" r="2" fill="#94a3b8"/><circle cx="23" cy="37" r="2" fill="#94a3b8"/>
        <path d="M6 56 L14 51 L22 54"/><circle cx="6" cy="56" r="2.4" fill="#94a3b8"/><circle cx="14" cy="51" r="2" fill="#94a3b8"/><circle cx="22" cy="54" r="2" fill="#94a3b8"/>
        <path d="M30 19 L58 30" stroke-dasharray="3 3"/>
        <path d="M30 37 L58 34" stroke-dasharray="3 3"/>
        <path d="M30 54 L58 38" stroke-dasharray="3 3"/>
        <path d="M58 30 L64 34 L57 38" fill="none"/>
      </g>
      <g fill="none" stroke="#cbd5e1" stroke-width="1.4" stroke-dasharray="3 3">
        <rect x="72" y="13" width="9" height="9" rx="1.5"/><rect x="84" y="13" width="9" height="9" rx="1.5"/><rect x="96" y="13" width="9" height="9" rx="1.5"/><rect x="108" y="13" width="9" height="9" rx="1.5"/><rect x="120" y="13" width="9" height="9" rx="1.5"/>
        <rect x="72" y="25" width="9" height="9" rx="1.5"/><rect x="84" y="25" width="9" height="9" rx="1.5"/><rect x="96" y="25" width="9" height="9" rx="1.5"/><rect x="108" y="25" width="9" height="9" rx="1.5"/><rect x="120" y="25" width="9" height="9" rx="1.5"/>
        <rect x="72" y="37" width="9" height="9" rx="1.5"/><rect x="84" y="37" width="9" height="9" rx="1.5"/><rect x="96" y="37" width="9" height="9" rx="1.5"/><rect x="108" y="37" width="9" height="9" rx="1.5"/><rect x="120" y="37" width="9" height="9" rx="1.5"/>
      </g>
    </svg>
    <span class="vr-chev" aria-hidden="true"></span>
  </div>
  <div class="vr-arrow" aria-hidden="true"></div>
  <div class="vr-step" tabindex="0" style="--vr-accent:#0d9488">
    <span class="vr-num">3</span>
    <div class="vr-body">
      <p class="vr-title">Task video fine-tune on your own data</p>
      <div class="vr-detail"><p>A rank-16 LoRA trained <em>purely on video prediction</em> of 32 episodes of our SO-101 data. Downstream action RMSE: 14.51&deg; &rarr; 13.06&deg;.</p>
      <p class="vr-warn">What not to do: joint video+action co-training collapsed to 24.37&deg;. Protect the video objective; train the policy on frozen features.</p></div>
    </div>
    <svg class="vr-illus" viewBox="0 0 150 62" aria-hidden="true">
      <g stroke="#0d9488" fill="none" stroke-width="2" stroke-linecap="round">
        <path d="M8 46 L18 30 L34 24"/>
        <circle cx="8" cy="46" r="3.4" fill="#0d9488"/><circle cx="18" cy="30" r="2.6" fill="#0d9488"/>
        <path d="M34 24 L39 20 M34 24 L39 28"/>
      </g>
      <rect x="42" y="22" width="8" height="8" rx="1.5" fill="#f59e0b"/>
      <g stroke="#64748b" fill="none" stroke-width="1.6">
        <line x1="50" y1="33" x2="60" y2="33"/>
        <path d="M60 29 L66 33 L60 37 z" fill="#64748b" stroke="none"/>
      </g>
      <g fill="#99f6e4" stroke="#0d9488" stroke-width="1">
        <rect x="72" y="13" width="9" height="9" rx="1.5"/><rect x="84" y="13" width="9" height="9" rx="1.5"/><rect x="96" y="13" width="9" height="9" rx="1.5"/><rect x="108" y="13" width="9" height="9" rx="1.5"/><rect x="120" y="13" width="9" height="9" rx="1.5"/>
        <rect x="72" y="25" width="9" height="9" rx="1.5"/><rect x="84" y="25" width="9" height="9" rx="1.5"/><rect x="96" y="25" width="9" height="9" rx="1.5"/><rect x="108" y="25" width="9" height="9" rx="1.5"/><rect x="120" y="25" width="9" height="9" rx="1.5"/>
        <rect x="72" y="37" width="9" height="9" rx="1.5"/><rect x="84" y="37" width="9" height="9" rx="1.5"/><rect x="96" y="37" width="9" height="9" rx="1.5"/><rect x="108" y="37" width="9" height="9" rx="1.5"/><rect x="120" y="37" width="9" height="9" rx="1.5"/>
      </g>
      <g fill="#f59e0b">
        <path d="M76.5 11 l3 3 l-3 3 l-3 -3 z"/>
        <path d="M100.5 23 l3 3 l-3 3 l-3 -3 z"/>
        <path d="M124.5 35 l3 3 l-3 3 l-3 -3 z"/>
      </g>
    </svg>
    <span class="vr-chev" aria-hidden="true"></span>
  </div>
  <div class="vr-arrow" aria-hidden="true"></div>
  <p class="vr-branch-label"><span>4 &mdash; adapt for observed-only (T=2) inference &middot; <em>proposed, runs in progress</em></span></p>
  <div class="vr-branches">
    <div class="vr-step" tabindex="0" style="--vr-accent:#7c3aed">
      <span class="vr-num">A</span>
      <div class="vr-body">
        <p class="vr-title">One-shot linear readout</p>
        <div class="vr-detail"><p>A per-position linear head predicts all 14 future latents in one shot from layer-20 T=2 features (plus a fresh LoRA). The bet: the features already encode the dynamics. Cheap &mdash; runs first as the baseline.</p></div>
      </div>
      <svg class="vr-illus vr-illus-sm" viewBox="0 0 150 62" aria-hidden="true">
        <g fill="#ddd6fe" stroke="#7c3aed" stroke-width="1.2">
          <rect x="4" y="22" width="12" height="18" rx="2"/>
          <rect x="19" y="22" width="12" height="18" rx="2"/>
        </g>
        <g stroke="#7c3aed" stroke-width="2" fill="none">
          <line x1="35" y1="31" x2="66" y2="31"/>
          <path d="M66 27 L72 31 L66 35 z" fill="#7c3aed" stroke="none"/>
        </g>
        <text x="50" y="24" text-anchor="middle" font-size="9" fill="#7c3aed" font-weight="700">1 step</text>
        <g fill="none" stroke="#7c3aed" stroke-width="1.2" opacity="0.85">
          <rect x="76" y="20" width="12" height="22" rx="2"/><rect x="90" y="20" width="12" height="22" rx="2"/><rect x="104" y="20" width="12" height="22" rx="2"/><rect x="118" y="20" width="12" height="22" rx="2"/><rect x="132" y="20" width="12" height="22" rx="2"/>
        </g>
      </svg>
      <span class="vr-chev" aria-hidden="true"></span>
    </div>
    <div class="vr-step" tabindex="0" style="--vr-accent:#0d9488">
      <span class="vr-num">B</span>
      <div class="vr-body">
        <p class="vr-title">World expert via prefix training</p>
        <div class="vr-detail"><p>A 115M diffusion expert denoises the 14 future latents while reading layer-20 T=2 features as prefix K/V &mdash; the exact interface the action expert consumes &mdash; jointly with a fresh LoRA. Bonus: it can generate video from the T=2 trunk.</p></div>
      </div>
      <svg class="vr-illus vr-illus-sm" viewBox="0 0 150 62" aria-hidden="true">
        <g fill="#99f6e4" stroke="#0d9488" stroke-width="1.2">
          <rect x="4" y="22" width="12" height="18" rx="2"/>
          <rect x="19" y="22" width="12" height="18" rx="2"/>
        </g>
        <g stroke="#0d9488" stroke-width="1.6" fill="none">
          <line x1="35" y1="31" x2="42" y2="31"/>
          <path d="M42 27 L47 31 L42 35 z" fill="#0d9488" stroke="none"/>
        </g>
        <rect x="50" y="17" width="28" height="28" rx="6" fill="none" stroke="#0d9488" stroke-width="1.8"/>
        <g stroke="#0d9488" stroke-width="1.6" fill="none">
          <path d="M69 26 A6.5 6.5 0 1 1 64 24"/>
          <path d="M62 20 L65 24 L60 26 z" fill="#0d9488" stroke="none"/>
        </g>
        <text x="64" y="56" text-anchor="middle" font-size="9" fill="#0d9488" font-weight="700">xN</text>
        <g stroke="#0d9488" stroke-width="1.6" fill="none">
          <line x1="82" y1="31" x2="89" y2="31"/>
          <path d="M89 27 L94 31 L89 35 z" fill="#0d9488" stroke="none"/>
        </g>
        <g fill="none" stroke="#0d9488" stroke-width="1.2" opacity="0.85">
          <rect x="98" y="20" width="11" height="22" rx="2"/><rect x="111" y="20" width="11" height="22" rx="2"/><rect x="124" y="20" width="11" height="22" rx="2"/><rect x="137" y="20" width="11" height="22" rx="2"/>
        </g>
      </svg>
      <span class="vr-chev" aria-hidden="true"></span>
    </div>
  </div>
  <div class="vr-arrow" aria-hidden="true"></div>
  <div class="vr-step" tabindex="0" style="--vr-accent:#d97706">
    <span class="vr-num">5</span>
    <div class="vr-body">
      <p class="vr-title">Cache features, train the policy</p>
      <div class="vr-detail"><p>One T=2 trunk forward per window (204 ms) &rarr; hash-validated feature cache &rarr; SmolExpert action head on top. The expert reaches its best score in under an hour.</p></div>
    </div>
    <svg class="vr-illus" viewBox="0 0 150 62" aria-hidden="true">
      <g fill="#fef3c7" stroke="#d97706" stroke-width="1.6">
        <path d="M6 18 a14 5 0 0 1 28 0 v22 a14 5 0 0 1 -28 0 z"/>
        <ellipse cx="20" cy="18" rx="14" ry="5"/>
      </g>
      <g stroke="#d97706" stroke-width="1.6" fill="none">
        <line x1="38" y1="30" x2="48" y2="30"/>
        <path d="M48 26 L54 30 L48 34 z" fill="#d97706" stroke="none"/>
      </g>
      <rect x="58" y="17" width="26" height="26" rx="6" fill="none" stroke="#d97706" stroke-width="1.8"/>
      <circle cx="66" cy="26" r="1.7" fill="#d97706"/><circle cx="76" cy="26" r="1.7" fill="#d97706"/>
      <path d="M64 34 q7 5 14 0" stroke="#d97706" stroke-width="1.6" fill="none"/>
      <g stroke="#d97706" stroke-width="1.6" fill="none">
        <line x1="88" y1="30" x2="95" y2="30"/>
        <path d="M95 26 L100 30 L95 34 z" fill="#d97706" stroke="none"/>
      </g>
      <path d="M104 40 C 114 40 112 20 122 20 C 132 20 130 34 140 30" stroke="#d97706" stroke-width="2" fill="none"/>
      <path d="M138 26 L144 29 L139 34 z" fill="#d97706"/>
    </svg>
    <span class="vr-chev" aria-hidden="true"></span>
  </div>
</div>

<style>
  .vam-recipe { margin: 1.6rem 0 2rem; display: flex; flex-direction: column; }
  .vam-recipe p { margin: 0; }
  .vam-recipe .vr-hint { font-size: 0.85em; opacity: 0.5; margin-bottom: 0.6rem; }
  .vam-recipe .vr-step {
    position: relative; display: flex; gap: 0.9rem; align-items: center;
    border: 1px solid rgba(128, 128, 128, 0.28); border-left: 3px solid var(--vr-accent);
    border-radius: 12px; padding: 0.8rem 1.1rem;
    background: rgba(128, 128, 128, 0.045);
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
    transition: border-color 0.18s ease, box-shadow 0.18s ease, transform 0.18s ease, background-color 0.18s ease;
    outline: none; cursor: default;
  }
  .vam-recipe .vr-step:hover, .vam-recipe .vr-step:focus, .vam-recipe .vr-step:focus-within {
    border-color: var(--vr-accent);
    background: color-mix(in srgb, var(--vr-accent) 6%, transparent);
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.09);
    transform: translateY(-1px);
  }
  .vam-recipe .vr-num {
    flex: none; align-self: flex-start; margin-top: 0.15rem;
    width: 1.8rem; height: 1.8rem; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.85em; font-weight: 700; color: #fff; background: var(--vr-accent);
  }
  .vam-recipe .vr-optional .vr-num { background: transparent; color: var(--vr-accent); border: 1.5px dashed var(--vr-accent); }
  .vam-recipe .vr-body { min-width: 0; flex: 1; align-self: flex-start; }
  .vam-recipe .vr-title { font-weight: 650; line-height: 1.8rem; letter-spacing: -0.01em; }
  .vam-recipe .vr-tag { font-weight: 400; font-size: 0.82em; opacity: 0.55; margin-left: 0.4rem; }
  .vam-recipe .vr-optional { border-style: dashed; border-left-style: dashed; }
  .vam-recipe .vr-optional .vr-title { opacity: 0.7; }
  .vam-recipe .vr-illus { flex: none; width: 150px; height: 62px; opacity: 0.95; }
  .vam-recipe .vr-illus-sm { width: 128px; height: 53px; }
  .vam-recipe .vr-chev {
    flex: none; width: 9px; height: 9px; margin-left: 0.3rem;
    border-right: 2px solid var(--vr-accent); border-bottom: 2px solid var(--vr-accent);
    transform: rotate(45deg); opacity: 0.55; transition: transform 0.25s ease;
  }
  .vam-recipe .vr-step:hover .vr-chev, .vam-recipe .vr-step:focus .vr-chev, .vam-recipe .vr-step:focus-within .vr-chev { transform: rotate(225deg) translate(-2px, -2px); }
  .vam-recipe .vr-detail { max-height: 0; opacity: 0; overflow: hidden; transition: max-height 0.3s ease, opacity 0.3s ease, margin-top 0.3s ease; font-size: 0.92em; line-height: 1.55; }
  .vam-recipe .vr-step:hover .vr-detail, .vam-recipe .vr-step:focus .vr-detail, .vam-recipe .vr-step:focus-within .vr-detail { max-height: 16rem; opacity: 0.92; margin-top: 0.35rem; }
  .vam-recipe .vr-detail p + p { margin-top: 0.45rem; }
  .vam-recipe .vr-warn { color: #dc2626; }
  .vam-recipe .vr-arrow { position: relative; width: 2px; height: 18px; margin: 2px auto; background: rgba(128, 128, 128, 0.45); }
  .vam-recipe .vr-arrow::after {
    content: ""; position: absolute; bottom: -1px; left: -3.5px;
    border-left: 4.5px solid transparent; border-right: 4.5px solid transparent;
    border-top: 6px solid rgba(128, 128, 128, 0.55);
  }
  .vam-recipe .vr-branch-label { display: flex; align-items: center; gap: 0.8rem; font-size: 0.88em; font-weight: 600; opacity: 0.75; margin: 0.15rem 0 0.6rem; }
  .vam-recipe .vr-branch-label::before, .vam-recipe .vr-branch-label::after { content: ""; flex: 1; height: 1px; background: rgba(128, 128, 128, 0.35); }
  .vam-recipe .vr-branches { display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem; }
  @media (max-width: 720px) { .vam-recipe .vr-illus { display: none; } }
  @media (max-width: 640px) { .vam-recipe .vr-branches { grid-template-columns: 1fr; } }
</style>

The compressed timeline, with the messy parts in the fold-outs.

**Days 1–2: audit and contract.** Pinned mimic-video's exact computation (frame contract, sigma semantics, layer-20 tap), the LTX architecture differences, and the SmolVLA baseline. Decided everything downstream would consume one frozen extractor contract so backbones stay swappable.

**Days 3–5: frozen features and baselines.** SmolVLA (a 450M VLA fine-tuned on the same data) set the reference at ~15.0/14.83°. Frozen Cosmos features + a from-scratch action decoder (World2Action-style, 499M) reached 14.51° after a connector ablation across pooling/token-selection variants. Two controls mattered: a *random-init* Cosmos backbone performed no worse than pretrained at small budgets (24.0° vs 23.9° at 30 min) — pretrained features only pay off at longer training — and a vision-randomized SmolVLA degraded to 17.43°, confirming the vision prior carries real signal.

<details>
<summary>Dead end: LoRA co-training (video + action jointly)</summary>

The obvious idea — backpropagate the action loss into LoRA adapters on the video backbone while also training video prediction — performed <em>worse</em> than frozen features: 24.37° vs 14.51°, with validation degrading over training. The action gradient appears to pull the representation away from the video-prediction manifold faster than the video loss can defend it, at least at this scale and data size (32 episodes). Decoupling the objectives — video LoRA first, then freeze, then action training — is what worked. This mirrors a broader lesson from the two weeks: <strong>protect the video-prediction objective; it is where the value lives.</strong>
</details>

**Days 6–9: LTX-2.5 and depth studies.** The 22B LTX reached 14.03° pooled / 13.84° unpooled — the best *frozen* backbone. Multi-depth probes (learned mixes over blocks 8–40) did not beat the single production tap. On the Cosmos side the picture was the same: layer 20 of 28 stays the extraction point.

**Day 9–10: swap the action head.** A controlled comparison on *identical* frozen Cosmos features: the from-scratch World2Action decoder (499M parameters, cross-attention in every block) had plateaued at 14.51° after ~24.6k steps. Replacing it with **SmolExpert** — SmolVLA's *pretrained* ~100M action expert plus a 2M connector, consuming the backbone tokens as prefix K/V — reached 14.14° within ~50 minutes and 13.65° converged. A 4× smaller head that trains faster and lands better; every result after this point uses SmolExpert.

<figure class="post-figure" data-zoomable>
  <img src="/assets/img/video-vam/smolexpert-vs-world2action.png" alt="Validation RMSE versus training steps and versus wall-clock hours for the two action heads on identical frozen Cosmos features">
  <figcaption>Identical frozen Cosmos features, two heads. The pretrained SmolExpert head (green) converges to 13.65° in ~2.5 hours; the from-scratch World2Action head (blue) needs an overnight run to reach 14.51°.</figcaption>
</figure>

**Days 10–12: video LoRA, and the win.** Fine-tuning Cosmos with a rank-16 LoRA purely on video prediction of the robot data (2 observed + 14 future latent frames, rectified-flow objective), then freezing it as the feature extractor: **13.06°**, the best result on this split. The side-by-side at the top of this post is that LoRA; here is what the *generic* model predicts on the same window before fine-tuning — it hallucinates plausible-but-wrong motion:

<figure class="post-figure">
  <video autoplay loop muted controls playsinline preload="metadata">
    <source src="/assets/video/video-vam/generic-ep19-side-by-side.mp4" type="video/mp4">
  </video>
  <figcaption>Base Cosmos-Predict2 2B, same conditioning frames, before LoRA fine-tuning (left: prediction, right: ground truth).</figcaption>
</figure>

**Days 12–13: the T=2 bet.** If the action head reads only the *observed* tokens' representation, do we need the 14 noisy future frames in the forward at all? Dropping them (16 latent frames → 2; 19,200 tokens → 2,400) gives 204 ms extraction — 5.7× faster — and costs 0.68° (13.74° vs 13.06°). That trade is the current frontier, and the subject of the open experiment in the last section.

<details>
<summary>Dead end: recovering the T=2 gap with multi-layer attention</summary>

First attempt to close the 0.68° gap: let the action expert attend to features from six depths (blocks 4–20) with a learned attention mixer instead of layer 20 alone, hoping intermediate layers retain dynamics that the truncated forward loses at the top. It did not recover the gap. The current hypothesis is that the loss is not about <em>which layer</em> you read but about severed computation: in the full forward, observed tokens iteratively exchange information with the future-token "scratchpad", and at T=2 that loop never runs.
</details>

## Making It Fast

The reference points, from BFL's own disclosures: FLUX-mimic reaches **<80 ms** to world representation on an RTX 5090 and **101 ms** end-to-end reaction time; the public non-FLUX mimic-video implementation was reported at **~1.3 s on an RTX 4090** by a maintainer. My unoptimized Cosmos path started at almost exactly that 1.3 s figure. Where it stands now:

| Path | Extraction / window | Notes |
|---|---:|---|
| Cosmos 2B, full 16-frame forward | 1,163 ms | production baseline |
| **Cosmos 2B, T=2 observed-only** | **204 ms** | 5.7×; −0.68° RMSE |
| LTX-2.5 FP8, CPU-offload, 8 latent frames | 1,243 ms | weight streaming dominates |
| LTX-2.5 FP8, CPU-offload, T=2 | 1,228 ms | token count irrelevant: 1.01× |
| LTX-2.5 INT4 weight-only, GPU-resident, T=2 | 780 ms | 22B model in 10.7 GiB VRAM |

Two engineering lessons generalize:

- **Know your bottleneck regime.** For the GPU-resident 2B model, latency scales with token count — cutting T=16 to T=2 gives the near-linear 5.7×. For the CPU-offloaded 22B model, latency is weight streaming — the same token cut gives 1.01×. The fix there is quantization: INT4 weight-only (torchao tinygemm) makes the 22B DiT fit a 24 GB card *resident*, for 1.57×.
- **Latency work is free until it isn't.** The T=2 speedup costs 0.68° of accuracy. The INT4 path still needs a feature-drift check and an expert retrain before it's usable. Every optimization above is paired with the quality bill it hasn't fully paid yet.

For real-time control the relevant number is chunk cadence: at 204 ms extraction + ~90 ms action decoding, a 3-second action chunk leaves comfortable margin, and real-time-chunking experiments (overlapping prediction with execution, LeRobot's guided RTC) already show seam RMSE reducible by 50–75% at the measured inference delays.

## Limitations

The big one: while verifying a normalization detail I found a **changepoint in the dataset at episode 24** — two joints flip the sign of their mean action, consistent with a re-mount partway through data collection. The train split (episodes 0–31) straddles the shift; the validation split (32–39) is entirely post-shift. Every RMSE above therefore partly measures generalization across a session shift, which plausibly explains why all methods plateau in a narrow 13.6–15.1° band. The *relative* comparisons share the confound and stand; the absolute numbers should be read with care. A cleanly re-recorded dataset with an interleaved split is the fix, and the next step.

Beyond the dataset: this is one task, one robot, 40 episodes, and proxy metrics end to end — no real-robot success rates yet. The SmolVLA baseline also used dataset-global normalization statistics (a leak in its favor), so the VAM-vs-SmolVLA gaps are, if anything, conservative.

## Open Frontier: Making T=2 as Good as T=16

The experiment running as this post goes live. The 0.68° gap between the observed-only forward and the full forward has a clean information-theoretic framing: in the full forward, the future tokens learn everything they know by attending to the observed tokens' K/V — so the observed representation is already a sufficient interface. What T=2 loses is the *computation* routed through the future-token scratchpad, plus a distribution shift the backbone never saw.

The fix under test — a **world expert**: keep the trunk at T=2, and train a small (115M) diffusion expert that denoises the 14 future VAE latents while reading the trunk's layer-20 features through the exact prefix-K/V interface the action expert uses, jointly with a fresh LoRA on the first 20 blocks. The expert cannot invent the future — all conditioning flows through the layer-20 interface — so the training pressure lands exactly on the representation the action head reads. As a bonus it can *generate videos* from the T=2 trunk, giving a direct visual eval against the full model: the trunk runs once, and only the small expert iterates over denoising steps. A near-free ablation (a per-position linear readout to the future latents) runs first as the baseline.

If it works, the target picture is: 204 ms feature extraction at 13.0° quality on a consumer GPU — and then, finally, deployment videos.

## What's Next

- World-expert and linear-readout results (runs queued).
- A clean dataset: fixed conditions, interleaved split, verified train/val statistics.
- Executed-prefix RMSE as the headline metric (what the robot actually performs under chunked replanning), alongside full-chunk RMSE.
- Real-robot rollouts with RTC seam blending — success rate is the metric that matters; everything above is proxy.

Experiments were tracked in a Weights & Biases project (private — happy to share details on request). The fine-tuned checkpoints and the dataset are public on [Hugging Face](https://huggingface.co/Orellius): the [Cosmos video LoRA](https://huggingface.co/Orellius/cube-out-of-box-cosmos-video-lora), the [Cosmos](https://huggingface.co/Orellius/cube-out-of-box-cosmos-pool2-smolexpert) and [LTX](https://huggingface.co/Orellius/cube-out-of-box-ltx-unpooled-smolexpert) SmolExpert policies, the [SmolVLA baselines](https://huggingface.co/Orellius/cube-out-of-box-smolvla-trainonly), and the [cube-out-of-box dataset](https://huggingface.co/datasets/Orellius/cube_out_of_box_v2). The stack is LeRobot plus a vendored mimic-video extractor; the SO-101 setup is described in the [robot learning setup post](/blog/2026/robot-learning-setup/).
