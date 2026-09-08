---
title: "Robot RL: From Demonstrations to Online Learning on a Real SO-101"
description: Seven months of building a robot learning pipeline — the timeline, the dead ends, and the progress. See also the companion technical overview post.
date: 2026-03-13
tags:
  - reinforcement-learning
  - robotics
authors:
  - name: Anton Wiehe
    url: https://notnanton.github.io
    affiliation: PHAROS Labs / AdaLab
featured: true
thumbnail: /assets/img/robot-matchbox-thumb.jpg
giscus: true
bibliography: 2026-03-13-robot-learning-setup.bib
---

## Overview
**March 2026 snapshot.** This diary describes the setup, observations, and plans as of March 2026; references to the current setup and next steps below belong to that time. For newer work, see [Video Models as Robot Policies](/blog/2026/video-models-as-robot-policies/).

This post covers about seven months of work on getting a real robot arm to learn manipulation tasks. It's meant as an honest account of the process — the dead ends, the debugging, and the things that eventually clicked — for anyone building a real-robot learning pipeline of their own.

The project is a two-person effort: I build it together with a friend at the [Attraktor](https://attraktor.org/) makerspace in Hamburg. "We" below means the two of us; "I" marks the parts I did on my own.

For a cleaner technical overview of the setup, techniques, and algorithms, see the [companion post](/blog/2026/robot-learning-setup/). This post is the lab diary.

The high-level goal is language-guided goal setting with quick learning of new tasks. The first concrete sub-goal was simple: grab a matchbox and place it somewhere. That turned out to be far from simple.

**State of the project in March 2026:**
- Imitation learning produces working pick-and-place policies on the real arm, though success is still inconsistent.
- In the simple proprioceptive RL experiments, the policy initially moved a lot, then reduced motion and held position during training. That is progress on a limited control task, not yet RL-based manipulation.
- Vision-based RL plus residual fine-tuning of the IL policies was the frontier then. See the [results table](/blog/2026/robot-learning-setup/#results-at-a-glance) in the setup post.

<figure class="post-figure">
  <video autoplay loop muted controls playsinline preload="metadata">
    <source src="/assets/video/robot-matchbox.webm" type="video/webm">
  </video>
  <figcaption>The SO-101 picking up a matchbox — one of our first successful imitation learning runs.</figcaption>
</figure>

Everything is built on [LeRobot](https://github.com/huggingface/lerobot) (HuggingFace's robot learning framework), with extensive modifications in our [fork](https://github.com/NotNANtoN/lerobot) on the `feat/human-reward-rl` branch to support joint-space control, delta action conversion, human reward feedback, current-based command attenuation, and various RL experiments.


## Hardware and Software

- **Robot**: SO-101 follower arm ("shabby"), five arm DOF plus one gripper DOF (six servo channels total), Feetech STS3215 servos
- **Teleoperation**: SO-101 leader arm for demonstration collection
- **Camera**: Single wrist-mounted camera (first-person view / FPV), 640x480, downsampled to 128x128 for policy input
- **Compute**: NVIDIA GPU for training, robot control on CPU
- **Framework**: LeRobot (fork), PyTorch, WandB for logging
- **Location**: [Attraktor](https://attraktor.org/) makerspace in Hamburg


## Timeline
### Aug-Sep 2025: Getting started with IL

We started at the Attraktor makerspace, trying to get imitation learning running on the matchbox task. The first few weeks were mostly fighting with the infrastructure:

- Dataset collection worked but had issues: weird channel ordering, corrupted recordings (40 episodes recorded but only 3 MP4 files locally), and manual version tagging
- The reward classifier trained but was unreliable — many false positives
- ACT training worked but evaluation on the robot failed due to hardcoded feature dimensions (6 position inputs, including the gripper, versus 12 when velocity features were included)
- We merged upstream LeRobot changes and hit more compatibility issues — no support for non-end-effector control in `gym_manipulator.py`

By September we had **SmolVLA** <d-cite key="shukor2025smolvla"></d-cite> and **ACT** <d-cite key="zhao2023act"></d-cite> training on 40 FPV episodes. Under 500 training steps nothing happened, but from 1000 steps both showed signs of reacting to the matchbox. SmolVLA at 10k steps actually managed to grasp once and place it, though it seemed random.

### Oct-Nov 2025: First RL attempts and overfitting

October brought the first RL attempts, which immediately ran into memory pressure from duplicate image buffers in the LeRobot replay path. We had to divide the dataset by 8 to make it fit, and even then the actor script crashed the machine.

In our SmolVLA setup, `n_action_steps` was set to 50: 50 actions executed per prediction, not a universal default. At our 10 Hz control rate that means about 5 seconds before replanning; at the 30 Hz recording rate, the same number of actions would span about 1.7 seconds. Reducing it to 5-10 made the policy much more reactive in our tests, at the cost of more frequent inference and replanning.

On the IL side, our early runs gave the impression that smaller batches were better. We trained ACT and SmolVLA at various settings:
- 200k steps SmolVLA lr=1e-4 bs=64 — overfitting, just repeated motions without looking
- 200k steps ACT lr=1e-5 bs=64 — poor performance
- 100k steps ACT lr=1e-5 bs=1 — first good FPV policy, actually grasped objects

But the story got more nuanced. In November, an exploratory comparison favored bs=64 ACT at 10k gradient steps over bs=1 at 100k: about 5/10 success versus 1/10. These are informal, recalled results, not a verified trial log or a controlled benchmark. That is 10x fewer gradient steps, not evidence of a 10x wall-clock speedup. The sample exposure also differed: 64 x 10k = 640k sampled examples versus 1 x 100k = 100k, including repeated examples. Training length and batch size were confounded, so this did not establish overfitting rather than batch size as the cause. In the SmolVLA run we took to 500k steps, later checkpoints also looked worse than those around 100k.

The setup post has the corresponding [training-loss curves](/blog/2026/robot-learning-setup/#training-details) for these runs.

<figure class="post-figure" data-zoomable>
  <img src="/assets/img/robot-rl/il-validation-loss-overfitting.png" alt="Validation loss for a SmolVLA run." loading="lazy" />
  <figcaption>Validation loss for one SmolVLA run. Loss bottoms out around 60k steps (vertical line marks the best checkpoint), then climbs past 80k. These are run-specific observations, not inevitable thresholds for a dataset of 40-100 demos. This plot motivated validation-based checkpoint selection, early stopping, and experiments with EMA weight averaging.</figcaption>
</figure>

This was also when the limited view from our wrist camera became a recurring problem. We suspected that temporal context (multiple past observations) would help, but on our compute budget we could only train with ~16 past frames — about half a second at the 30 Hz recording rate, not the 10 Hz control rate. Trying 2-5 seconds of context was a hypothesis, not an established requirement or a fundamental limit of FPV policies.

We also set up WandB for shared logging and added validation loss tracking using actual inference logic rather than relying only on training loss. Comparing architectures still requires the same inference-based metric, preprocessing, evaluation data, and action horizon; their raw losses are not automatically comparable.

### Nov 2025: The pretrained weights discovery

By late November I was genuinely frustrated — after months of work, we were still struggling to get consistent performance. Then came a painful realization: we had been training SmolVLA completely from scratch — not even using the pretrained VLM weights. The PEFT policy was producing outputs close to the default calibration position, which tipped us off. Months of SmolVLA experiments had been running without the pretrained backbone.

After fixing this, we recorded a much larger dataset (nearly 100 episodes of cube sorting with 4 cubes) and added early stopping with EMA weight averaging.

### Jan 2026: Renavigating

After a break, we came back with clearer goals:
- Debug the context-16 observation window (it wasn't working correctly)
- Get RL running with a pretrained policy and human-in-the-loop rewards
- Decide between training a reward function vs. pure human feedback

### Jan-Feb 2026: First RL that works

Late January brought progress on simple RL tasks. Key findings:
- The ResNet reward classifier still wasn't reliable — too many false positives
- We added motor-current readings as a reward signal (negative reward for high current, except gripper). Current is a load proxy here, not a direct measurement of joint torque or contact force
- We implemented UMAP visualization of trajectories to understand what the policy was exploring

We also studied the **ResFiT** paper (Ankile et al. 2025) <d-cite key="ankile2025resfit"></d-cite> in detail and identified concrete things to implement: n-step returns, higher UTD ratio (we were at ~0.2 effective when training at 2Hz but operating at 10Hz), delayed policy updates, and DrQ augmentations.

On February 13 — a long night session — we mapped out the full ResFiT architecture in detail: freeze the IL policy, predict its action for each sample, let SAC learn small residual corrections bounded to 20% of action range, use a shallow ViT with DrQ augmentations for the RL critic, layer norm only in the critic, update actor every 2-8 critic steps, and sample 50/50 from expert and newly collected data. Their results with ACT on real hardware were compelling: ~1,000 demonstrations for the base ACT policy, then 134 online RL rollouts (~15 minutes) improving that task from 14% to 64% success.

I also attended the Mannheim RL Workshop 2026, which solidified my thinking around algorithm design: separating the core algorithm from stabilization techniques (e.g., replay buffer is fundamental, target network is a stabilizer). Met the Darmstadt RL group (Jan Peters) and picked up the idea of using future state entropy as an exploration signal.

### Feb-Mar 2026: Minimal RL and compliance

The distributed actor-learner architecture was too complex for debugging. So I wrote `minimal_rl.py` — a single-process SAC script that bypasses all the infrastructure and talks directly to the robot.

I started with the simplest possible problem: hold the home position. Reward = negative distance to home, six position inputs including the gripper, no camera. The policy initially moved a lot, then appeared to learn to reduce motion and hold position; the selected run below shows this change over roughly 800 steps. That was encouraging adaptive behavior, not robust manipulation or a general convergence rate. The servos already hold commanded positions, so a zero-delta-action baseline is still needed to quantify what the learner adds.

<figure class="post-figure" data-zoomable>
  <img src="/assets/img/robot-rl/rl-base-learning-curves.png" alt="A selected position-holding RL run on the real robot." loading="lazy" />
  <figcaption>A selected position-holding run on the real robot. Five panels from minimal_rl.py: reward (negative distance to home) approaches zero over roughly 800 steps, Q-values settle around -200, the current/load-proxy trace stays moderate, the control loop holds 6-8 Hz, and the entropy coefficient alpha decreases. The trace labelled torque comes from motor current; its physical units have not been verified. The decreasing alpha alone does not show that the policy became deterministic or explored usefully. This run exercised the robot communication, reward loop, and learner together on physical hardware.</figcaption>
</figure>

Then I added a motor-current penalty as a second objective. We observed position holding with a low current penalty, not a measurement of minimum force. The experiments also revealed that RL exploration can put real strain on the servos. That directly motivated the next step.

In early March, I integrated a **low-gain PI controller** to make position commands less aggressive. I supplemented it with motor-current-based command attenuation and action limits. These heuristics reduced commanded changes and observed loads in supervised experiments, but they are not calibrated force control and do not guarantee compliance, safety, or recovery. Current is only a load proxy, and the servos still use their own position controllers.

I also tried a weight-projection step inspired by **XQC** <d-cite key="palenicek2026xqc"></d-cite> (projecting linear layer weights to unit sphere after each gradient step) together with delayed policy updates (TD3-style, update actor every 4 critic updates). The runs looked more stable, but I did not isolate either change, so this remains a stability hypothesis. This addition to minimal SAC is not a faithful implementation of the full XQC algorithm.

<figure class="post-figure" data-zoomable>
  <img src="/assets/img/robot-rl/rl-weight-norm-learning-curves.png" alt="Position holding with weight projection, delayed policy updates, and a motor-current penalty." loading="lazy" />
  <figcaption>A selected run with weight projection, delayed policy updates, and a motor-current penalty. Six panels: total reward, position reward, the penalty labelled torque, Q-values, motor-current/load-proxy readings, and control frequency. The current penalty stays near zero; that does not establish minimum force, and the current trace has no verified physical units here. Q-values around -5 to -8 rather than -200 are not evidence of better calibration: several settings and the reward changed between runs. The reward has intermittent dips, and position holding emerges over roughly 1600 steps in this recording, not on a universal convergence schedule.</figcaption>
</figure>

The next step we had in mind then was adding vision to these simple tasks (position holding with camera input) before moving to the real target: RL-based improvement of the pick-and-place policies.


## The Pipeline

The full pipeline has three stages. In practice, I've been mostly working with the minimal RL script for stage 3.

### Step 1: Demonstration Collection

Leader-follower teleoperation records joint positions, wrist-camera images, and timestamps as LeRobot episodes. The hardware and collection setup are described in the [setup post](/blog/2026/robot-learning-setup/#teleoperation).

Controls during recording:
- **Right Arrow**: save episode
- **Left Arrow**: discard and re-record
- **Escape**: stop

### Step 2: Reward Labeling and Classifier Training

A post-processing script labels successful terminal frames for a reward classifier. Its limitations led us to combine it with human feedback and motor-current signals; see [Reward Models](/blog/2026/robot-learning-setup/#reward-models).

### Step 3: Reinforcement Learning

Two options:

**Distributed (LeRobot's actor-learner architecture)**: Actor runs on the robot and the learner trains SAC over gRPC. Its additional components made failures harder to isolate than in the single-process script.

**Minimal single-process (`minimal_rl.py`)**: Everything runs in one script and was my main tool for rapid iteration in March 2026. For reward construction and RL details, see [Reinforcement Learning](/blog/2026/robot-learning-setup/#reinforcement-learning).


## Current Setup — Minimal RL

The script I was actively using in March 2026 was `minimal_rl.py` — a ~1000-line single-process SAC implementation that talks directly to the robot hardware. Configuration at that point:

| Parameter | Value | Notes |
|-----------|-------|-------|
| FPS | 10 | Control loop frequency |
| Learning rate | 3e-4 | Adam, same for all networks |
| Batch size | 64 | |
| Buffer size | 10,000 | Simple deque |
| Discount | 0.99 | |
| Tau | 0.005 | Soft target update rate |
| UTD ratio | 4 | Critic updates per env step |
| Warmup | 50 steps | Random actions before learning |
| Episode length | 200 steps | Max steps per episode |
| Policy delay | 4 | Actor update every 4 critic updates |
| Weight norm | On | XQC-inspired projection to unit sphere, not full XQC |
| Batch norm | Off | Fragile with batch_size=1 inference |
| Reward norm | Off | Running mean/std — still buggy |
| Mixed precision | bf16 | On CUDA |

The architecture is simple: a 256-dim encoder (Linear + LayerNorm + ReLU), a 2-layer actor MLP outputting tanh-squashed Gaussian actions, and a twin-critic with 2-layer MLPs. The six position inputs include the five arm joints and the gripper, normalized to [-1, 1]. Camera input can be enabled (and was used for all supervised learning with ACT/SmolVLA), but is off by default in the minimal RL script for the simple proprioceptive tasks we've been debugging with. Getting RL to work with visual observations on the real robot was still an active next step then.

At that point, the reward was: `reward = -distance_to_home / 100 - torque_penalty`. Despite its name in the code, `torque_penalty` is a motor-current-based load penalty, not a direct torque or force measurement. It uses a softplus form: `penalty = scale * log(1 + exp(0.5 * (sum_I_squared/1000 - 10)))`.

On Ctrl+C, the script automatically generates:
- Learning curves (reward, Q-values, motor-current proxy labelled torque, entropy temperature)
- Heatmaps showing critic Q-values and actor responses across the joint space
- UMAP visualizations of visited states using policy distribution distances


## Key Technical Decisions
The main design choices were driven by safety, sample efficiency, and practical debugging:

- **Delta actions + interpolation:** Use local action changes and split oversized demonstration transitions to limit commanded changes during supervised RL exploration; see [Action Spaces](/blog/2026/robot-learning-setup/#action-spaces).
- **Current penalties + command limits:** Penalize high currents and attenuate commands as load-reduction heuristics, without guaranteeing recovery toward home; see [Reward Models](/blog/2026/robot-learning-setup/#reward-models) and [Controller and Compliance](/blog/2026/robot-learning-setup/#controller-and-compliance).
- **Weight projection:** Try to improve optimization stability in low-data real-robot training; see [XQC](/blog/2026/robot-learning-setup/#xqc).
- **Human reward feedback:** Provide immediate shaping when a visual reward model is unreliable; see [Human-in-the-Loop Rewards](/blog/2026/robot-learning-setup/#human-in-the-loop-rewards).


## What Worked and What Didn't

**Worked well:**
- Leader-follower teleoperation — reliable and intuitive
- Small ACT policies (bs=1, 100k steps) — our first real success on the matchbox task, though later comparisons found better configurations
- Delta action interpolation — essential for bridging IL demos and RL
- Minimal single-process RL — dramatically faster iteration than distributed
- Weight projection + delayed policy updates — runs looked more stable when tried together; individual effects were not isolated
- Motor-current penalty and command attenuation — helped reduce observed loads in supervised experiments, not a guarantee against self-damage
- Low-gain PI controller and action limits — reduced commanded changes; not calibrated force control or a safety guarantee

**Still in progress:**
- Reward classifier — performance is task-dependent; works well for clear visual contrasts, needs tuning for ambiguous scenes
- Batch normalization in RL — implemented but not yet properly tested on the real robot due to batch_size=1 inference complications
- Reward normalization — running mean/std needs more work to stabilize in early training
- RL with vision — supervised learning with camera images works (that's how ACT/SmolVLA operate), but we haven't gotten RL to work well with visual observations yet

**Didn't work well:**
- SmolVLA from scratch — we accidentally trained without pretrained weights for months
- Long training on small IL datasets — later checkpoints looked worse, consistent with overfitting; the comparisons were not controlled
- Distributed actor-learner — was too complex for our initial debugging; the minimal single-process script made iteration easier


## What's Next

**Immediate (the next weeks, as planned in March 2026):**
- **ResFiT (Residual fine-tuning)**: Freeze a pretrained ACT/SmolVLA policy and learn small residual corrections with SAC; we have the full architecture mapped out.
- **XQC on real hardware**: Beyond the minimal script's projection step, our separate categorical-critic and batch-norm implementation had been tried in simulation. Single-observation inference already used the actor's evaluation mode for BN running statistics; validating the complete policy on the robot remained open. See the [implementation details](/blog/2026/robot-learning-setup/#xqc).
- **Visual RL tasks**: Move beyond joint-only observations to camera-conditioned policies for real manipulation (matchbox grasp, cube sorting).

**Medium-term:**
- **Hierarchical RL**: A proposed two-level Director architecture where a high-level manager picks discrete goal codes every K steps and a low-level worker executes. We hoped this might help with sparse rewards: the worker would train with dense goal-reaching rewards while the manager handled sparse task rewards. We had sketched out hindsight relabeling and competence-based curriculum learning, but had not demonstrated that this solves our sparse-reward problem.
- **Temporal context for FPV**: Our wrist-camera view often left out useful information. We wanted to explore xLSTM as an alternative to transformers for temporal processing, potentially replacing or augmenting SmolVLA's architecture. Bounded recurrent state could make streaming inference cheaper, but it is not infinite usable context; whether longer history would help our task remained to be tested.
- **Better reward signals**: Dense rewards from visual features (color histograms, optical flow), learned reward models with active querying.

**Longer-term:**
- **Mannheim RL Workshop 2027**: Targeting a presentation or poster on the real-robot RL pipeline.
- **Language-conditioned goal setting**: The original high-level goal — tell the robot what to do in natural language and have it learn quickly.
