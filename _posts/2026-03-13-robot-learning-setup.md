---
layout: distill
title: "Real-Robot Learning on a Budget: Setup, Stack, and Techniques"
description: A technical overview of my robot learning setup — hardware, software, imitation learning, reward models, and reinforcement learning on a real SO-101 arm.
tags: reinforcement-learning robotics imitation-learning
giscus_comments: true
date: 2026-03-13
featured: true
bibliography: 2026-03-13-robot-learning-setup.bib

authors:
  - name: Anton Wiehe
    url: "https://notnanton.github.io"
    affiliations:
      name: PHAROS Labs / AdaLab

toc:
  - name: Introduction
  - name: Hardware
    subsections:
      - name: Robot Arm
      - name: Sensors
      - name: Teleoperation
  - name: Software Stack
  - name: Imitation Learning
    subsections:
      - name: ACT
      - name: SmolVLA
      - name: Training Details
  - name: Reward Models
  - name: Reinforcement Learning
    subsections:
      - name: SAC
      - name: Action Spaces
      - name: Torque-Aware Safety
      - name: Human-in-the-Loop Rewards
      - name: XQC
  - name: Experimental Settings
  - name: Open Problems

---

## Introduction

This post describes the technical setup behind my robot learning experiments. It's written as a standalone reference for anyone who wants to understand the components, the techniques, and how they fit together.

The goal is to teach a low-cost robot arm to perform manipulation tasks with minimal human effort. The approach combines imitation learning from demonstrations with reinforcement learning for online improvement. Everything runs on a single-GPU workstation and a 120-euro robot arm.

For the narrative of how we got here — the timeline, the dead ends, the breakthroughs — see the [companion post]({% post_url 2026-03-13-robot-rl-experiments %}).


## Hardware

### Robot Arm

The **SO-101** is an open-source 6-DOF robot arm with a parallel-jaw gripper, built from 3D-printed parts and Feetech STS3215 servo motors. It communicates over a single USB-to-serial connection using the Feetech protocol.

Key properties:
- 6 joints: pan, tilt, elbow (2 servos in differential), roll, gripper
- Position control at up to 30 Hz
- Current (torque) sensing via `Present_Current` register — used for reward shaping and safety
- Low cost (~120 EUR for parts), but mechanically imprecise — backlash and compliance vary across units

We call ours "shabby" for a reason. The imprecision is actually a feature for research: methods that work on this hardware should transfer easily to better arms.

### Sensors

**Camera.** A single USB camera mounted on the wrist (first-person view / FPV), capturing 640x480 RGB at 30 fps. Images are downsampled to 128x128 for policy input (or 64x64 for some RL experiments).

FPV-only is a deliberate constraint. Most prior work uses fixed overhead cameras that see the entire workspace. With FPV, the robot can only see what's in front of its gripper, making tasks like "find the object" require memory or exploration. This is harder but more general — a mobile robot wouldn't have an overhead camera either.

**Proprioception.** Joint positions (6 DOF) and motor currents (6 channels), both read from the servo bus at the control frequency.

### Teleoperation

Demonstrations are collected via a **leader-follower** setup: a second SO-101 arm (the leader) is physically moved by a human. The follower arm mirrors the leader's joint positions in real time. Joint positions and camera images are recorded as a LeRobot dataset at the control FPS.

This approach produces natural, kinematically consistent demonstrations without any inverse kinematics or motion planning. The downside is that the human can't feel what the follower is touching (no force feedback).


## Software Stack

Everything builds on [**LeRobot**](https://github.com/huggingface/lerobot), HuggingFace's open-source robot learning framework. We maintain a fork with extensions for:

- **Joint-space control** in the gym environment (upstream only supported end-effector / Cartesian)
- **Delta action conversion** with interpolation for large movements
- **Reward classifier training and evaluation** from demonstration datasets
- **Human-in-the-loop reward feedback** via keyboard during RL
- **Torque penalty and safety layers** using motor current sensing
- **XQC policy** (Cross-Q with Corrections — distributional SAC variant)
- **Combined Experience Replay** mixing recent transitions with replay buffer
- **Rerun integration** for real-time visualization of images, actions, and metrics

Training and logging use **PyTorch** and **Weights & Biases**. For rapid prototyping, we also have `minimal_rl.py`, a standalone single-process SAC script that bypasses the distributed architecture and talks to the robot directly.


## Imitation Learning

We train visuomotor policies from demonstrations using two architectures.

### ACT

**Action Chunking with Transformers** <d-cite key="zhao2023act"></d-cite> is a CVAE-based policy. The encoder is a ResNet backbone processing camera images and a transformer that fuses visual and proprioceptive tokens. The decoder predicts a *chunk* of future actions (typically 50-100 timesteps) in one forward pass.

Key properties for our setup:
- ResNet backbone uses pretrained ImageNet weights
- Works well with small datasets (40-100 demonstrations)
- Predicts absolute joint positions
- Training is straightforward: L1 reconstruction loss on action chunks
- Default `n_action_steps=50` means the robot executes ~5 seconds of open-loop control per prediction at 10 Hz. Reducing to 5-10 gives more reactive behavior.

### SmolVLA

**SmolVLA** <d-cite key="allal2025smolvla"></d-cite> adapts a vision-language model (VLM) backbone for robot control. The architecture reuses a pretrained SmolVLM-2 and fine-tunes it to output motor commands.

Key properties:
- Pretrained VLM backbone — this is critical. Training from scratch (which we accidentally did for months) produces near-random policies.
- Designed for 512x512 image input, though we often train at 128x128 for speed.
- More parameter-efficient fine-tuning via LoRA/PEFT is possible but still under development.
- Better sample efficiency on some tasks due to the pretrained visual features, but more sensitive to hyperparameters than ACT.

### Training Details

| Setting | ACT | SmolVLA |
|---------|-----|---------|
| Learning rate | 1e-5 | 1e-4 |
| Batch size | 8-64 | 64 |
| Training steps | 10k-100k | 15k-200k |
| Image resolution | dataset native | 128x128-512x512 |
| Pretrained backbone | ResNet (ImageNet) | SmolVLM-2 |
| Action chunk length | 50-100 | 50 |

Findings from our experiments:
- **Batch size matters more than training steps.** ACT at bs=64 for 10k steps matched or exceeded bs=1 at 100k steps (4-5/10 vs. 1/10 success rate on cube sorting), reaching the same loss ~10x faster.
- **Overfitting is the main enemy.** SmolVLA clearly overfits past 100k steps on our dataset sizes (40-100 episodes). Validation loss tracking and early stopping with EMA weight averaging are essential.
- **Loss correlates with performance**, at least within the same architecture. This isn't obvious — the training loss is reconstruction error on action chunks, not task success.
- **FPV-only is limited by context.** With a single wrist camera and `n_obs_steps=1`, the robot has no memory. Increasing context to 16 frames (~1.6s) helps but isn't enough for tasks requiring search. We likely need 2-5 seconds or a fundamentally different architecture (recurrent, xLSTM).

{% include figure.html path="assets/img/robot-rl/il-training-loss-comparison.png" caption="WandB training loss (action chunk MSE) on the cube sorting task. Three ACT runs (orange, green, red) at different learning rates and batch sizes plateau around loss 1.5-2.0, while SmolVLA (teal) converges to near zero. The gap comes from the pretrained VLM backbone: SmolVLA starts with strong visual features, so the policy head only needs to learn the action mapping. All runs use the same 40-episode FPV dataset." class="img-fluid rounded z-depth-1" zoomable=true %}

{% include figure.html path="assets/img/robot-rl/il-validation-loss-overfitting.png" caption="Validation loss (computed with actual inference logic, not teacher-forced) for a SmolVLA training run on cube sorting. Loss decreases until around 60k steps, then starts climbing and becomes noisy past 80k. The vertical line marks the best checkpoint. This motivated adding early stopping with EMA weight averaging to all our training scripts. With only 40-100 demos, overfitting is the main failure mode." class="img-fluid rounded z-depth-1" zoomable=true %}


## Reward Models

For RL, we need a reward signal. We use three approaches, often in combination.

**Learned reward classifier.** A ResNet-10 binary classifier trained on demonstration images. The last N frames of each successful demonstration are labeled `reward=1.0`, everything else `reward=0.0`. The classifier predicts task success from a single camera image.

Performance is task-dependent. For tasks with clear visual contrasts (matchbox on a bag vs. empty surface) it reaches ~0.7 confidence on true positives. For cluttered scenes with distracting objects, false positive rates increase. We use `pos_weight` to handle class imbalance in the training loss.

**Torque-based rewards.** Motor currents are a cheap proxy for undesirable contact. High currents usually mean the robot is pushing against something (the table, itself, a joint limit). The penalty is:

$$\text{penalty} = w \cdot \sum_{i} I_i^2 / 1000$$

where \\(I_i\\) is the current on motor \\(i\\) and \\(w\\) is a tunable weight. The gripper motor can be excluded since gripping *should* produce current.

**Human-in-the-loop rewards.** During live RL training, a human operator presses number keys (0-9) to provide graded rewards, or minus (-) for punishment. This is surprisingly effective for shaping early behavior and doesn't require any visual model.


## Reinforcement Learning

### SAC

The core RL algorithm is **Soft Actor-Critic** <d-cite key="haarnoja2018sac"></d-cite>, an off-policy maximum-entropy method. It jointly trains:

- An **actor** (policy network): outputs a squashed Gaussian distribution over delta joint actions
- A **twin critic**: two Q-networks that estimate expected return, taking the minimum to prevent overestimation
- A **temperature** \\(\alpha\\): automatically tuned to target a desired entropy level

We use standard SAC with several additions:
- **Delayed policy updates** (TD3-style): the actor updates every 4 critic updates, which helps stability
- **Weight normalization**: after each gradient step, Linear layer weights (except output heads) are projected to unit norm. This keeps the effective learning rate constant.
- **Update-to-data (UTD) ratio of 4**: 4 gradient steps per environment step. This is important because environment steps are expensive (real-time on hardware), so we want to extract as much learning as possible from each transition.

{% include figure.html path="assets/img/robot-rl/rl-base-learning-curves.png" caption="SAC on real hardware: position holding with 6 joint angles as input. Top panel: reward (negative squared distance to home position) converges to near zero within 800 steps. Second panel: Q-value estimates stabilize around -150 to -200. Third: torque readings stay moderate (0.5-1.5 range). Fourth: control loop frequency holds at 6-8 Hz. Bottom: the entropy coefficient alpha decays from 0.20 to 0.12 as the policy becomes more deterministic. This was our first successful end-to-end RL run on the real robot." class="img-fluid rounded z-depth-1" zoomable=true %}

{% include figure.html path="assets/img/robot-rl/rl-weight-norm-learning-curves.png" caption="Same position holding task with weight normalization and torque penalty added. Now tracking six panels: total reward, position reward component, torque penalty (near zero, meaning the robot holds position with minimal force), Q-values (much smaller magnitude around -5 to -8 vs. -200 without weight norm), torque, and control frequency. The reward signal is noisier with periodic dips from exploration, but Q-values are better calibrated and the torque penalty stays flat. Alpha decays more slowly, suggesting the policy maintains useful exploration longer. Run converges in about 1600 steps." class="img-fluid rounded z-depth-1" zoomable=true %}

### Action Spaces

**Absolute vs. delta.** Demonstrations are recorded as absolute joint positions, but the RL policy outputs delta actions: small increments relative to the current position. This makes exploration local and safe.

$$a_{\text{target}} = q_{\text{current}} + \text{tanh}(\pi(s)) \cdot \Delta_{\max}$$

where \\(\Delta_{\max} = \text{action\_scale\_per\_s} / \text{fps}\\) (default: 50/10 = 5.0 position-units per step).

**Demo conversion.** When seeding the replay buffer from demonstrations, absolute-position transitions are converted to deltas. If any delta exceeds the action scale, the transition is split into N sub-transitions with linearly interpolated proprioception and zero-order-hold images. Without this, the policy would be asked to learn impossibly large single-step jumps.

### Torque-Aware Safety

The robot is fragile. We implement two safety layers:

**Proactive.** Every delta action is clamped to \\(\pm 5°\\) per step, preventing large sudden movements regardless of what the policy outputs.

**Reactive.** Motor currents are read each step. When current exceeds a threshold, the commanded action is attenuated. Movements back toward the home position are exempted from attenuation so the robot can recover from loaded positions.

These safety measures run at the environment level and are transparent to the policy. The policy sees only that certain actions produce less movement than expected.

### Human-in-the-Loop Rewards

During live RL, a human watches the robot and presses keys to provide reward:

| Key | Reward |
|-----|--------|
| 0-9 | 0.0 to 0.9 (graded) |
| - | -0.5 (punishment) |

The human reward is added to whatever automated signal exists (torque penalty, classifier). This makes it possible to start training from scratch without any reward model — the human provides the learning signal until an automated reward becomes reliable.

### XQC

**Cross-Q with Corrections** <d-cite key="bhatt2024crossq"></d-cite> is an extension of SAC that we've implemented but not yet deployed on the real robot. The key ideas:

- **No target networks.** Instead, batch normalization is applied to a *joined* batch of current and next states during the critic forward pass. This ensures consistent normalization statistics and removes the need for a separate target network and its soft update.
- **Distributional critic (C51).** Instead of predicting a single Q-value, the critic outputs a categorical distribution over returns (51 atoms). The loss is a categorical cross-entropy against a projected Bellman target. This captures uncertainty and tends to learn faster.
- **Weight normalization.** After each optimizer step, all hidden-layer weights are projected to the unit sphere, keeping the effective learning rate stable regardless of weight magnitude.

The implementation is complete and tested, but batch normalization with batch_size=1 during inference (when the robot takes a single action) requires careful handling of BN running statistics. This is the main blocker for real-robot deployment.


## Experimental Settings

We've been working bottom-up, starting with the simplest possible RL problems on real hardware before adding complexity.

**Imitation learning task: pick and place.** Grab a cube (or matchbox) and place it at a target position. Trained from 40-100 teleoperated demonstrations using FPV camera input. This is the main benchmark for our IL policies (ACT, SmolVLA).

**RL task 1: Position holding (joints only).** The simplest possible RL problem. The reward is the negative squared distance from the current joint positions to a home position. No camera, just 6 joint angles as input. The policy learns to stay still despite exploration noise. Converges in ~800 steps. This validated that our SAC implementation, reward loop, and robot communication all work correctly.

**RL task 2: Torque minimization (joints only).** Same as above but with a torque penalty added: the policy is penalized for high motor currents. This teaches the robot to hold position while using minimal force. The task revealed that unconstrained exploration can put dangerous pressure on the servos, which led us to implement the PI compliance controller and the reactive safety layer.

**Next RL tasks (planned):**
- Position holding with vision — same reward, but the policy receives camera images instead of (or in addition to) joint positions. A stepping stone to visual RL.
- Vision-based pick and place — combining the IL task with online RL improvement via ResFiT or direct fine-tuning.


## Open Problems

These are the questions I'm currently working on or planning to tackle, roughly in priority order.

**Residual fine-tuning (ResFiT).** Freeze a pretrained IL policy, train a small residual correction on top with SAC. This approach keeps the structure of demonstrations while allowing RL to fix systematic errors. Prior work shows 13% to 64% success rate improvements with 15 minutes of real-world RL <d-cite key="ankile2025resfit"></d-cite>.

**RL with visual observations.** All current RL experiments use only proprioception (joint positions). Extending to camera-conditioned policies requires a visual encoder in the actor and critic, data augmentation (DrQ-style), and likely much more data. The IL policies already use vision, so the question is how to transfer that capability to the RL setting.

**Temporal context for FPV.** A single frame from a wrist camera doesn't contain enough information for many tasks. Options include: stacking past frames as transformer context, using recurrent architectures (xLSTM looks promising for its constant memory and efficient inference), or maintaining an explicit spatial memory.

**Hierarchical RL.** For tasks with sparse rewards and long horizons, flat RL is unlikely to work. We're sketching a two-level architecture: a high-level manager that selects discrete goal codes every K steps, and a low-level worker that executes. The worker gets dense goal-reaching rewards; the manager handles sparse task rewards. Hindsight relabeling of the manager's goals provides additional training signal.

**Better reward models.** The current binary classifier is too coarse. Directions include: dense rewards from visual features (color histograms, optical flow), active querying to improve the model where it's uncertain, and multi-modal rewards that combine vision with proprioception and torque.

