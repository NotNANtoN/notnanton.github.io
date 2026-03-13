---
layout: distill
title: "Robot RL: From Demonstrations to Online Learning on a Real SO-101"
description: A report on seven months of building a robot learning pipeline — what I tried, what broke, and what finally worked.
tags: reinforcement-learning robotics
giscus_comments: true
date: 2026-03-13
featured: true

authors:
  - name: Anton Wiehe
    url: "https://notnanton.github.io"
    affiliations:
      name: PHAROS Labs / AdaLab

toc:
  - name: Overview
  - name: Hardware and Software
  - name: Timeline
    subsections:
      - name: "Aug-Sep 2025: Getting started with IL"
      - name: "Oct-Nov 2025: First RL attempts and overfitting"
      - name: "Nov 2025: The pretrained weights discovery"
      - name: "Jan 2026: Renavigating"
      - name: "Jan-Feb 2026: First RL that works"
      - name: "Feb-Mar 2026: Minimal RL and compliance"
  - name: The Pipeline
    subsections:
      - name: "Step 1: Demonstration Collection"
      - name: "Step 2: Reward Labeling and Classifier Training"
      - name: "Step 3: Reinforcement Learning"
  - name: Current Setup — Minimal RL
  - name: Key Technical Decisions
    subsections:
      - name: Delta Actions and Interpolation
      - name: Torque Penalties and Safety
      - name: Weight Normalization
      - name: Human Reward Feedback
  - name: What Worked and What Didn't
  - name: What's Next

---

## Overview

This post covers about seven months of work on getting a real robot arm to learn manipulation tasks. It's meant as an honest account of the process — the dead ends, the debugging, and the things that eventually clicked. The goal is to give my professor and potential collaborators a clear picture of where the project stands, what infrastructure exists, and where students could plug in.

The high-level goal is language-guided goal setting with quick learning of new tasks. The first concrete sub-goal was simple: grab a matchbox and place it somewhere. That turned out to be far from simple.

Everything is built on [LeRobot](https://github.com/huggingface/lerobot) (HuggingFace's robot learning framework), with extensive modifications to support joint-space control, delta action conversion, human reward feedback, torque-aware safety, and various RL improvements.


## Hardware and Software

- **Robot**: SO-101 follower arm ("shabby"), 6 DOF + gripper, Feetech STS3215 servos
- **Teleoperation**: SO-101 leader arm for demonstration collection
- **Camera**: Single front-view (FPV) camera, 640x480, downsampled to 128x128 for policy input
- **Compute**: NVIDIA GPU for training, robot control on CPU
- **Framework**: LeRobot (fork), PyTorch, WandB for logging
- **Location**: [Attraktor](https://attraktor.org/) makerspace in Hamburg


## Timeline

### Aug-Sep 2025: Getting started with IL

We started at the Attraktor makerspace, trying to get imitation learning running on the matchbox task. The first few weeks were mostly fighting with the infrastructure:

- Dataset collection worked but had issues: weird channel ordering, corrupted recordings (40 episodes recorded but only 3 MP4 files locally), and manual version tagging
- The reward classifier trained but was unreliable — many false positives
- ACT training worked but evaluation on the robot failed due to hardcoded feature dimensions (6 joints expected, but 12 with velocity features)
- We merged upstream LeRobot changes and hit more compatibility issues — no support for non-end-effector control in `gym_manipulator.py`

By September we had SmolVLA and ACT training on 40 FPV episodes. Under 500 training steps nothing happened, but from 1000 steps both showed signs of reacting to the matchbox. SmolVLA at 10k steps actually managed to grasp once and place it, though it seemed random.

### Oct-Nov 2025: First RL attempts and overfitting

October brought the first RL attempts, which immediately ran into a memory problem: the LeRobot replay buffer stored images 3 times (480x640x3 for 10k frames = 34 GB). We had to divide the dataset by 8 to make it fit, and even then the actor script crashed the machine.

We also discovered that SmolVLA defaults to executing 50 actions per prediction (~5 seconds of open-loop control at 10 Hz). Reducing `n_action_steps` to 5-10 made the policy much more reactive, and was essential for tasks requiring closed-loop behavior.

On the IL side, we discovered that batch size matters a lot. We trained ACT and SmolVLA at various settings:
- 200k steps SmolVLA lr=1e-4 bs=64 — overfitting, just repeated motions without looking
- 200k steps ACT lr=1e-5 bs=64 — poor performance
- 100k steps ACT lr=1e-5 bs=1 — first good FPV policy, actually grasped objects

But the story got more nuanced. In November, we did more systematic comparisons and found that bs=64 ACT at 10k steps performed comparably or better than bs=1 at 100k (4-5/10 success vs. 1/10), reaching the same loss much faster. The key insight: loss does seem to correlate with performance, but both overfit eventually. SmolVLA trained for 500k steps was clearly overfit past 100k.

This was also when we recognized a fundamental limitation of FPV-only control: with a single wrist camera, the robot can't see the full scene. We need temporal context (multiple past observations), but on our compute budget we could only train with ~16 past frames — about half a second. We probably need 2-5 seconds of context.

We also set up WandB for shared logging and added validation loss tracking to detect overfitting properly, using actual inference logic rather than training loss to make different architectures comparable.

### Nov 2025: The pretrained weights discovery

By late November I was genuinely frustrated — five months in and struggling to get consistent performance. Then came a painful realization: we had been training SmolVLA completely from scratch — not even using the pretrained VLM weights. The PEFT policy was producing outputs close to the default calibration position, which tipped us off. Months of SmolVLA experiments had been running without the pretrained backbone.

After fixing this, we recorded a much larger dataset (nearly 100 episodes of cube sorting with 4 cubes), added proper early stopping with EMA weight averaging, and started contributing fixes back upstream (validation loss tracking, notes on pretrained weight defaults).

### Jan 2026: Renavigating

After a break, we came back with clearer goals:
- Debug the context-16 observation window (it wasn't working correctly)
- Get RL running with a pretrained policy and human-in-the-loop rewards
- Decide between training a reward function vs. pure human feedback

We also started contributing back to LeRobot: validation loss PR, PEFT training fixes, and various compatibility patches.

### Jan-Feb 2026: First RL that works

Late January was the breakthrough. Key findings:
- The ResNet reward classifier still wasn't reliable — too many false positives
- We added torque reading as a reward signal (negative reward for high motor current, except gripper)
- We implemented UMAP visualization of trajectories to understand what the policy was exploring

We also studied the ResFiT paper (Ankile et al. 2025) in detail and identified concrete things to implement: n-step returns, higher UTD ratio (we were at ~0.2 effective when training at 2Hz but operating at 10Hz), delayed policy updates, and DrQ augmentations.

On February 13 — a long night session — we mapped out the full ResFiT architecture in detail: freeze the IL policy, predict its action for each sample, let SAC learn small residual corrections bounded to 20% of action range, use a shallow ViT with DrQ augmentations for the RL critic, layer norm only in the critic, update actor every 2-8 critic steps, and sample 50/50 from expert and newly collected data. Their results with ACT on real hardware were compelling: 1000 demos plus 134 online RL episodes (15 minutes) improved a bimanual pick-and-place from 13% to 64% success.

I also attended the Mannheim RL Workshop 2026, which solidified my thinking around algorithm design: separating the core algorithm from stabilization techniques (e.g., replay buffer is fundamental, target network is a stabilizer). Met the Darmstadt RL group (Jan Peters) and picked up the idea of using future state entropy as an exploration signal.

### Feb-Mar 2026: Minimal RL and compliance

The distributed actor-learner architecture was too complex for debugging. So I wrote `minimal_rl.py` — a single-process SAC script that bypasses all the infrastructure and talks directly to the robot.

The results were immediate: with a simple "stay at home position" reward (negative distance to home, plus torque penalty), the policy learned in about 800 steps. Learning rate wasn't very sensitive — anything from 1e-2 to 3e-4 worked, with 1e-4 converging fastest.

I also added weight normalization (projecting linear layer weights to unit sphere after each gradient step, XQC-style) and delayed policy updates (TD3-style, update actor every 4 critic updates). Both helped stability.

In early March I integrated a compliance controller — a PI controller that makes the robot give way to resistance, with a safety layer that reads motor currents and attenuates commands when torque is too high.


## The Pipeline

The full pipeline has three stages. In practice, I've been mostly working with the minimal RL script for stage 3.

### Step 1: Demonstration Collection

Leader-follower teleoperation: physically move the leader arm and the follower mirrors. Episodes are saved as LeRobot datasets with joint positions, camera images, and timestamps.

Controls during recording:
- **Right Arrow**: save episode
- **Left Arrow**: discard and re-record
- **Escape**: stop

I typically collect 20-50 demonstrations per task. The data is stored as absolute joint positions at the recording FPS (usually 10 or 30).

### Step 2: Reward Labeling and Classifier Training

A post-processing script marks the last N frames of each successful episode as `reward=1.0`, everything else as `reward=0.0`. A ResNet-10 classifier trains on these labeled images.

Honest assessment: the reward classifier has been hit-or-miss. It works OK for clear visual differences (cube in hand vs. not), but produces many false positives when the scene is ambiguous. I'm currently working more with human reward feedback and torque-based rewards instead.

### Step 3: Reinforcement Learning

Two options:

**Distributed (LeRobot's actor-learner architecture)**: Actor runs on the robot, learner trains SAC, they communicate over gRPC. This is the full-featured version with offline replay buffer seeding, reward classifiers, and human intervention via keyboard. It works but is complex to debug.

**Minimal single-process (`minimal_rl.py`)**: Everything in one script. Currently my main tool for rapid iteration. See next section.


## Current Setup — Minimal RL

The script I'm actively using is `minimal_rl.py` — a ~1000-line single-process SAC implementation that talks directly to the robot hardware. Current default configuration:

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
| Weight norm | On | XQC-style projection to unit sphere |
| Batch norm | Off | Fragile with batch_size=1 inference |
| Reward norm | Off | Running mean/std — still buggy |
| Mixed precision | bf16 | On CUDA |

The architecture is simple: a 256-dim encoder (Linear + LayerNorm + ReLU), a 2-layer actor MLP outputting tanh-squashed Gaussian actions, and a twin-critic with 2-layer MLPs. Joint positions are the input (6 DOF, normalized to [-1, 1]). Camera can be enabled but is off by default for the simple tasks.

Reward is currently: `reward = -distance_to_home / 100 - torque_penalty`. The torque penalty uses the sum of squared motor currents.

On Ctrl+C, the script automatically generates:
- Learning curves (reward, Q-values, torque, entropy temperature)
- Heatmaps showing critic Q-values and actor responses across the joint space
- UMAP visualizations of visited states using policy distribution distances


## Key Technical Decisions

### Delta Actions and Interpolation

Demonstrations are recorded as absolute joint positions, but the RL policy operates in delta space. The conversion:

```
delta = (action - current_position) / action_scale
```

`action_scale = action_scale_per_s / fps` (default: 50/10 = 5.0 per step). The policy outputs in [-1, 1], which maps to a maximum of 5 position-units per step.

When converting offline demos, if any delta exceeds the action scale, the transition is split into N sub-transitions with interpolated proprioception and zero-order-hold images. This was essential — without it, the policy was trying to learn impossible single-step jumps.

### Torque Penalties and Safety

Motor currents are read via `Present_Current` from the Feetech bus. The torque penalty uses a softplus:

```
penalty = scale * log(1 + exp(0.5 * (sum_I_squared/1000 - 10)))
```

There's also a safety layer that clamps max delta per step (5 degrees) and attenuates commands when current is already high. Movements toward home are exempted from current attenuation so the robot can recover from gravity-loaded positions.

### Weight Normalization

After each gradient step, all Linear layer weights (except output heads) are projected onto the unit sphere. This is from the XQC paper and helps with training stability by keeping the effective learning rate constant. I have a full XQC implementation (distributional C51 critic with categorical cross-entropy loss, batch normalization with joined forward passes) ready but haven't deployed it on the real robot yet — the batch norm is fragile with batch_size=1 during inference.

### Human Reward Feedback

During RL training, number keys 0-9 provide graded rewards (0.0 to 0.9), minus key gives punishment (-0.5). This works alongside or instead of the reward classifier. The rewards are added to whatever other reward signal exists (torque penalty, classifier) within the same step.


## What Worked and What Didn't

**Worked well:**
- Leader-follower teleoperation — reliable and intuitive
- Small ACT policies (bs=1, 100k steps) — first real success on the matchbox task
- Delta action interpolation — essential for bridging IL demos and RL
- Minimal single-process RL — dramatically faster iteration than distributed
- Weight normalization — noticeable stability improvement
- Torque penalty — effective at preventing self-damage during exploration
- PI compliance controller — makes the robot safe to be around

**Didn't work well / still struggling:**
- Reward classifier — too many false positives, unreliable as sole reward
- SmolVLA from scratch — we accidentally trained without pretrained weights for months
- Large batch sizes for IL — overfit faster than expected
- Batch normalization in RL — works in theory (XQC) but batch_size=1 inference is painful
- Reward normalization — running mean/std is unstable in early training
- Distributed actor-learner — correct but hard to debug, minimal script was the fix


## What's Next

**Immediate (next weeks):**
- **ResFiT (Residual fine-tuning)**: Freeze a pretrained ACT/SmolVLA policy, learn small residual corrections with SAC. The paper (Ankile et al. 2025) shows improvements from 13% to 64% success rate with only 15 minutes of real-world RL. We have the full architecture mapped out.
- **XQC on real hardware**: The distributional critic (C51) and batch norm architecture is implemented and tested in simulation. Need to solve the inference-mode BN issue and deploy.
- **Visual RL tasks**: Move beyond joint-only observations to camera-conditioned policies for real manipulation (matchbox grasp, cube sorting).

**Medium-term:**
- **Hierarchical RL**: A two-level Director architecture where a high-level manager picks discrete goal codes every K steps and a low-level worker executes. This addresses the sparse reward problem in real-robot tasks. The worker trains with dense goal-reaching rewards while the manager handles sparse task rewards. We've sketched out hindsight relabeling and competence-based curriculum learning.
- **Temporal context for FPV**: Single-frame FPV is fundamentally limited. Exploring xLSTM as an alternative to transformers for temporal processing — constant memory, efficient inference, potentially infinite context. Could replace or augment SmolVLA's architecture.
- **Better reward signals**: Dense rewards from visual features (color histograms, optical flow), learned reward models with active querying.

**Longer-term:**
- **Mannheim RL Workshop 2026**: Targeting a presentation or poster on the real-robot RL pipeline.
- **Language-conditioned goal setting**: The original high-level goal — tell the robot what to do in natural language and have it learn quickly.

**Contributions back to LeRobot (PRs in progress or planned):**
- Validation loss tracking with proper inference-time evaluation
- PEFT fine-tuning fixes and documentation for pretrained weight defaults
- Joint-space control in gym_manipulator
- Various compatibility patches between versions

**Where students could plug in:**
- ResFiT implementation and real-robot evaluation (needs someone comfortable with PyTorch and willing to be hands-on with hardware)
- Reward model improvements — the classifier is the weakest link
- Hierarchical RL / goal-conditioned policies — mostly a research question, well-suited for a thesis
- xLSTM or other temporal architectures for FPV-only control
- Multi-camera or depth perception (we've been FPV-only, adding a second camera would change a lot)
