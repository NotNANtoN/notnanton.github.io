---
title: "Robot RL: From Demonstrations to Online Learning on a Real SO-101"
description: Seven months of building a robot learning pipeline — the timeline, the dead ends, and the breakthroughs. See also the companion technical overview post.
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
This post covers about seven months of work on getting a real robot arm to learn manipulation tasks. It's meant as an honest account of the process — the dead ends, the debugging, and the things that eventually clicked — for anyone building a real-robot learning pipeline of their own.

The project is a two-person effort: I build it together with a friend at the [Attraktor](https://attraktor.org/) makerspace in Hamburg. "We" below means the two of us; "I" marks the parts I did on my own.

For a cleaner technical overview of the setup, techniques, and algorithms, see the [companion post](/blog/2026/robot-learning-setup/). This post is the lab diary.

The high-level goal is language-guided goal setting with quick learning of new tasks. The first concrete sub-goal was simple: grab a matchbox and place it somewhere. That turned out to be far from simple.

**State of the project:**
- Imitation learning produces working pick-and-place policies on the real arm.
- Reinforcement learning works end-to-end on the real robot for simple proprioceptive tasks (position holding, torque minimisation).
- Vision-based RL plus residual fine-tuning of the IL policies is the current frontier. See the [results table](/blog/2026/robot-learning-setup/#results-at-a-glance) in the setup post.

<figure class="post-figure">
  <video autoplay loop muted controls playsinline preload="metadata">
    <source src="/assets/video/robot-matchbox.webm" type="video/webm">
  </video>
  <figcaption>The SO-101 picking up a matchbox — one of our first successful imitation learning runs.</figcaption>
</figure>

Everything is built on [LeRobot](https://github.com/huggingface/lerobot) (HuggingFace's robot learning framework), with extensive modifications in our [fork](https://github.com/NotNANtoN/lerobot) on the `feat/human-reward-rl` branch to support joint-space control, delta action conversion, human reward feedback, torque-aware safety, and various RL improvements.


## Hardware and Software

- **Robot**: SO-101 follower arm ("shabby"), 6 DOF + gripper, Feetech STS3215 servos
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
- ACT training worked but evaluation on the robot failed due to hardcoded feature dimensions (6 joints expected, but 12 with velocity features)
- We merged upstream LeRobot changes and hit more compatibility issues — no support for non-end-effector control in `gym_manipulator.py`

By September we had **SmolVLA** <d-cite key="shukor2025smolvla"></d-cite> and **ACT** <d-cite key="zhao2023act"></d-cite> training on 40 FPV episodes. Under 500 training steps nothing happened, but from 1000 steps both showed signs of reacting to the matchbox. SmolVLA at 10k steps actually managed to grasp once and place it, though it seemed random.

### Oct-Nov 2025: First RL attempts and overfitting

October brought the first RL attempts, which immediately ran into a memory problem: the LeRobot replay buffer stored images 3 times (480x640x3 for 10k frames = 34 GB). We had to divide the dataset by 8 to make it fit, and even then the actor script crashed the machine.

We also discovered that SmolVLA defaults to executing 50 actions per prediction (~5 seconds of open-loop control at 10 Hz). Reducing `n_action_steps` to 5-10 made the policy much more reactive, and was essential for tasks requiring closed-loop behavior.

On the IL side, our early runs gave the impression that smaller batches were better. We trained ACT and SmolVLA at various settings:
- 200k steps SmolVLA lr=1e-4 bs=64 — overfitting, just repeated motions without looking
- 200k steps ACT lr=1e-5 bs=64 — poor performance
- 100k steps ACT lr=1e-5 bs=1 — first good FPV policy, actually grasped objects

But the story got more nuanced. In November, a systematic comparison showed that bs=64 ACT at 10k steps reached better results about 10x faster than bs=1 at 100k (4-5/10 success vs. 1/10). The early impression that bs=1 was better came from training runs of different lengths; the actual culprit was overfitting from too many steps on a small dataset, not batch size itself. SmolVLA trained for 500k steps was clearly overfit past 100k.

The setup post has the corresponding [training-loss curves](/blog/2026/robot-learning-setup/#training-details) for these runs.

<figure class="post-figure" data-zoomable>
  <img src="/assets/img/robot-rl/il-validation-loss-overfitting.png" alt="Validation loss for a SmolVLA run." loading="lazy" />
  <figcaption>Validation loss for a SmolVLA run. Loss bottoms out around 60k steps (vertical line marks the best checkpoint), then climbs steadily past 80k. With only 40-100 demos, overfitting is inevitable if you train too long. This plot is what convinced us to add early stopping with EMA weight averaging to every training script.</figcaption>
</figure>

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

We also studied the **ResFiT** paper (Ankile et al. 2025) <d-cite key="ankile2025resfit"></d-cite> in detail and identified concrete things to implement: n-step returns, higher UTD ratio (we were at ~0.2 effective when training at 2Hz but operating at 10Hz), delayed policy updates, and DrQ augmentations.

On February 13 — a long night session — we mapped out the full ResFiT architecture in detail: freeze the IL policy, predict its action for each sample, let SAC learn small residual corrections bounded to 20% of action range, use a shallow ViT with DrQ augmentations for the RL critic, layer norm only in the critic, update actor every 2-8 critic steps, and sample 50/50 from expert and newly collected data. Their results with ACT on real hardware were compelling: ~1,000 demonstrations for the base ACT policy, then 134 online RL rollouts (~15 minutes) improving that task from 14% to 64% success.

I also attended the Mannheim RL Workshop 2026, which solidified my thinking around algorithm design: separating the core algorithm from stabilization techniques (e.g., replay buffer is fundamental, target network is a stabilizer). Met the Darmstadt RL group (Jan Peters) and picked up the idea of using future state entropy as an exploration signal.

### Feb-Mar 2026: Minimal RL and compliance

The distributed actor-learner architecture was too complex for debugging. So I wrote `minimal_rl.py` — a single-process SAC script that bypasses all the infrastructure and talks directly to the robot.

I started with the simplest possible problem: hold the home position. Reward = negative distance to home, joints only as input, no camera. The policy learned in about 800 steps. Learning rate wasn't very sensitive — anything from 1e-2 to 3e-4 worked, with 1e-4 converging fastest.

<figure class="post-figure" data-zoomable>
  <img src="/assets/img/robot-rl/rl-base-learning-curves.png" alt="The first successful RL run on the real robot." loading="lazy" />
  <figcaption>The first successful RL run on the real robot. Five panels from our minimal_rl.py script: reward (negative distance to home) converges to near zero within 800 steps, Q-values stabilize around -200, torque stays moderate, the control loop holds 6-8 Hz, and the entropy coefficient alpha decays as the policy commits to staying still. Simple task, but this validated the entire pipeline: SAC, robot communication, reward loop, and safety layer all working together.</figcaption>
</figure>

Then I added torque minimization as a second objective. The policy learned to hold position with minimal force, but the experiments revealed that RL exploration can put real strain on the servos. That directly motivated the next step.

In early March, I integrated a **PI compliance controller**. Instead of the robot fighting every external force to reach a target, the PI controller allows for "give," making the robot physically compliant. I supplemented this with a reactive safety layer that reads motor currents (torque) and attenuates commands when the load is too high. This move away from a simple position-command interface made experimentation much safer and allowed for longer RL runs without hardware failure.

I also added weight normalization (projecting linear layer weights to unit sphere after each gradient step, **XQC**-style <d-cite key="palenicek2026xqc"></d-cite>) and delayed policy updates (TD3-style, update actor every 4 critic updates). Both helped stability.

<figure class="post-figure" data-zoomable>
  <img src="/assets/img/robot-rl/rl-weight-norm-learning-curves.png" alt="Position holding with weight normalization and torque penalty." loading="lazy" />
  <figcaption>Position holding with weight normalization and torque penalty. Now six panels: total reward, position reward, torque penalty (near zero, meaning the robot holds with minimal force), Q-values (much better calibrated at -5 to -8 vs. -200 without weight norm), torque readings, and control frequency. The reward is noisier due to periodic exploration dips, but the Q-values are realistic and the torque penalty stays flat. Converges in about 1600 steps.</figcaption>
</figure>

The next step is adding vision to these simple tasks (position holding with camera input) before moving to the real target: RL-based improvement of the pick-and-place policies.


## The Pipeline

The full pipeline has three stages. In practice, I've been mostly working with the minimal RL script for stage 3.

### Step 1: Demonstration Collection

Leader-follower teleoperation records joint positions, wrist-camera images, and timestamps as LeRobot episodes. The hardware and collection setup are described in the [setup post](/blog/2026/robot-learning-setup/#teleoperation).

Controls during recording:
- **Right Arrow**: save episode
- **Left Arrow**: discard and re-record
- **Escape**: stop

### Step 2: Reward Labeling and Classifier Training

A post-processing script labels successful terminal frames for a reward classifier. Its limitations led us to combine it with human feedback and torque signals; see [Reward Models](/blog/2026/robot-learning-setup/#reward-models).

### Step 3: Reinforcement Learning

Two options:

**Distributed (LeRobot's actor-learner architecture)**: Actor runs on the robot and the learner trains SAC over gRPC. It is the full-featured version, but complex to debug.

**Minimal single-process (`minimal_rl.py`)**: Everything runs in one script and is currently my main tool for rapid iteration. For reward construction and RL details, see [Reinforcement Learning](/blog/2026/robot-learning-setup/#reinforcement-learning).


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

The architecture is simple: a 256-dim encoder (Linear + LayerNorm + ReLU), a 2-layer actor MLP outputting tanh-squashed Gaussian actions, and a twin-critic with 2-layer MLPs. Joint positions are the input (6 DOF, normalized to [-1, 1]). Camera input can be enabled (and was used for all supervised learning with ACT/SmolVLA), but is off by default in the minimal RL script for the simple proprioceptive tasks we've been debugging with. Getting RL to work with visual observations on the real robot is an active next step.

Reward is currently: `reward = -distance_to_home / 100 - torque_penalty`. The torque penalty uses the canonical softplus form: `penalty = scale * log(1 + exp(0.5 * (sum_I_squared/1000 - 10)))`.

On Ctrl+C, the script automatically generates:
- Learning curves (reward, Q-values, torque, entropy temperature)
- Heatmaps showing critic Q-values and actor responses across the joint space
- UMAP visualizations of visited states using policy distribution distances


## Key Technical Decisions
The main design choices were driven by safety, sample efficiency, and practical debugging:

- **Delta actions + interpolation:** Use local action changes and split oversized demonstration transitions so RL exploration stays safe and learnable; see [Action Spaces](/blog/2026/robot-learning-setup/#action-spaces).
- **Torque penalties + safety:** Penalize high currents and attenuate hazardous commands while preserving recovery toward home; see [Reward Models](/blog/2026/robot-learning-setup/#reward-models) and [Controller and Compliance](/blog/2026/robot-learning-setup/#controller-and-compliance).
- **Weight normalization:** Stabilize optimization in low-data real-robot training; see [XQC](/blog/2026/robot-learning-setup/#xqc).
- **Human reward feedback:** Provide immediate shaping when a visual reward model is unreliable; see [Human-in-the-Loop Rewards](/blog/2026/robot-learning-setup/#human-in-the-loop-rewards).


## What Worked and What Didn't

**Worked well:**
- Leader-follower teleoperation — reliable and intuitive
- Small ACT policies (bs=1, 100k steps) — our first real success on the matchbox task, though later comparisons found better configurations
- Delta action interpolation — essential for bridging IL demos and RL
- Minimal single-process RL — dramatically faster iteration than distributed
- Weight normalization — noticeable stability improvement
- Torque penalty — effective at preventing self-damage during exploration
- PI compliance controller — makes the robot safe to be around

**Still in progress:**
- Reward classifier — performance is task-dependent; works well for clear visual contrasts, needs tuning for ambiguous scenes
- Batch normalization in RL — implemented but not yet properly tested on the real robot due to batch_size=1 inference complications
- Reward normalization — running mean/std needs more work to stabilize in early training
- RL with vision — supervised learning with camera images works (that's how ACT/SmolVLA operate), but we haven't gotten RL to work well with visual observations yet

**Didn't work well:**
- SmolVLA from scratch — we accidentally trained without pretrained weights for months
- Long training on small IL datasets — overfitting made later checkpoints and comparisons look worse
- Distributed actor-learner — functionally correct but too complex to debug; the minimal single-process script was the fix


## What's Next

**Immediate (next weeks):**
- **ResFiT (Residual fine-tuning)**: Freeze a pretrained ACT/SmolVLA policy and learn small residual corrections with SAC; we have the full architecture mapped out.
- **XQC on real hardware**: The distributional critic (C51) and batch norm architecture is implemented and tested in simulation. Need to solve the inference-mode BN issue and deploy.
- **Visual RL tasks**: Move beyond joint-only observations to camera-conditioned policies for real manipulation (matchbox grasp, cube sorting).

**Medium-term:**
- **Hierarchical RL**: A two-level Director architecture where a high-level manager picks discrete goal codes every K steps and a low-level worker executes. This addresses the sparse reward problem in real-robot tasks. The worker trains with dense goal-reaching rewards while the manager handles sparse task rewards. We've sketched out hindsight relabeling and competence-based curriculum learning.
- **Temporal context for FPV**: Single-frame FPV is fundamentally limited. Exploring xLSTM as an alternative to transformers for temporal processing — constant memory, efficient inference, potentially infinite context. Could replace or augment SmolVLA's architecture.
- **Better reward signals**: Dense rewards from visual features (color histograms, optical flow), learned reward models with active querying.

**Longer-term:**
- **Mannheim RL Workshop 2027**: Targeting a presentation or poster on the real-robot RL pipeline.
- **Language-conditioned goal setting**: The original high-level goal — tell the robot what to do in natural language and have it learn quickly.

**Contributions back to LeRobot (PRs in progress or planned):**
- [Validation loss tracking, early stopping, and checkpoint cleanup](https://github.com/huggingface/lerobot/pull/2633) — open upstream PR
- PEFT fine-tuning fixes and documentation for pretrained weight defaults
- Joint-space control in gym_manipulator
- Various compatibility patches between versions
