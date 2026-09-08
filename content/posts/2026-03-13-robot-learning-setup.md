---
title: "Real-Robot Learning on a Budget: Setup, Stack, and Techniques"
description: A technical overview of our robot learning setup — hardware, software, imitation learning, reward models, and reinforcement learning on a real SO-101 arm.
date: 2026-03-13
tags:
  - reinforcement-learning
  - robotics
  - imitation-learning
authors:
  - name: Anton Wiehe
    url: https://notnanton.github.io
    affiliation: PHAROS Labs / AdaLab
featured: true
thumbnail: /assets/img/robot-cube-thumb.jpg
giscus: true
bibliography: 2026-03-13-robot-learning-setup.bib
---

## Introduction
> **Historical snapshot: March 2026.** References to our current setup, results, and planned work below describe that period, not the latest project status. For the later direction, see [Video Models as Robot Policies](/blog/2026/video-models-as-robot-policies/).

This post describes the technical setup behind our robot learning experiments — a two-person project I run with a friend at the [Attraktor](https://attraktor.org/) makerspace in Hamburg. It's written as a standalone reference for anyone who wants to understand the components, the techniques, and how they fit together.

The goal is to teach a low-cost robot arm to perform manipulation tasks with minimal human effort. Our manipulation results come from imitation learning; separate, simple RL experiments are groundwork toward online improvement of those policies. Everything runs on a single-GPU workstation and a 120-euro robot arm.

<figure class="post-figure" data-zoomable>
  <img src="/assets/img/robot-cube.jpg" alt="The SO-101 arm at our workspace, ready for cube sorting — one of the main benchmark tasks for our imitation learning experiments." loading="lazy" />
  <figcaption>The SO-101 arm at our workspace, ready for cube sorting — one of the main benchmark tasks for our imitation learning experiments.</figcaption>
</figure>

For the narrative of how we got here — the timeline, the dead ends, the breakthroughs — see the [companion post](/blog/2026/robot-rl-experiments/).

## Results at a Glance
Here are the headline results from the current setup:

| Task | Policy | Training | Result |
|------|--------|----------|--------|
| Matchbox pick-and-place | ACT (bs=1, 100k steps) | 40 FPV demos | observed successful grasps |
| Cube sorting | ACT (bs=64, 10k steps) | 40 FPV demos | about 5/10 successes (informal, recalled count) |
| Cube sorting | ACT (bs=1, 100k steps) | 40 FPV demos | 1/10 successes in the earlier exploratory trials |
| Position holding (joints only) | SAC | online, real robot | less motion; reward near zero at ~800 env steps in this run |
| Position holding + current-based penalty | SAC + weight norm | online, real robot | approximate reward plateau at ~1600 env steps in this run |

The cube-sorting figures summarize small exploratory trial sets. The **about 5/10** figure is an informal, recalled count, not an exact recorded success rate; the earlier **1/10** is retained as a reference, not a controlled comparison. The RL step counts describe the observed runs, not general convergence rates.


## Hardware
### Robot Arm

The **SO-101** is an open-source robot with **5 arm DOF plus 1 gripper DOF** (6 servos total), built from 3D-printed parts and Feetech STS3215 servo motors, with a parallel-jaw gripper. It communicates over a single USB-to-serial connection using the Feetech protocol.

Key properties:
- 6 controlled joints: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`
- Position control at up to 30 Hz
- Motor-current readings via the `Present_Current` register — used as load proxies for reward shaping and action attenuation, not calibrated torque or contact-force measurements
- Low cost (~120 EUR for parts), but mechanically imprecise — backlash and compliance vary across units

We call ours "shabby" for a reason. The imprecision makes it an interesting testbed for robustness. Transfer to better arms remains an open question because their dynamics, sensing, and control interfaces differ.

### Sensors

**Camera.** A single USB camera mounted on the wrist (first-person view / FPV), capturing 640x480 RGB at 30 fps. Images are downsampled to 128x128 for policy input (or 64x64 for some RL experiments).

FPV-only is a deliberate constraint. Many manipulation setups use external cameras with a broader workspace view. With FPV, the robot can only see what's in front of its gripper, making tasks like "find the object" require memory or exploration. This lets us study limited-view control relevant to some mobile setups.

**Proprioception.** Six position channels (five arm joints plus the gripper) and six motor-current channels, both read from the servo bus at the control frequency. The gripper position is not a sixth arm angle; channel units depend on the control interface and normalization.

### Teleoperation

Demonstrations are collected via a **leader-follower** setup: a second SO-101 arm (the leader) is physically moved by a human. The follower arm mirrors the leader's joint positions in real time. Joint positions and camera images are recorded as a LeRobot dataset at 30 Hz, while the RL control loop runs at 10 Hz.

This approach produces natural, kinematically consistent demonstrations without any inverse kinematics or motion planning. The downside is that the human can't feel what the follower is touching (no force feedback).


## Software Stack
Everything builds on [**LeRobot**](https://github.com/huggingface/lerobot), HuggingFace's open-source robot learning framework. We maintain a [fork](https://github.com/NotNANtoN/lerobot) on the `feat/human-reward-rl` branch with extensions for:

- **Joint-space control** in the gym environment (the upstream environment path we used then supported end-effector / Cartesian control)
- **Delta action conversion** with interpolation for large movements
- **Reward classifier training and evaluation** from demonstration datasets
- **Human-in-the-loop reward feedback** via keyboard during RL
- **Current-based load penalty and action-limiting heuristics** using motor-current readings
- **XQC-inspired policy** (local distributional SAC variant with batch and weight normalization; see the implementation caveats below)
- **Combined Experience Replay** mixing recent transitions with replay buffer
- **Rerun integration** for real-time visualization of images, actions, and metrics

Training and logging use **PyTorch** and **Weights & Biases**. For rapid prototyping, we also have `minimal_rl.py`, a standalone single-process SAC script that bypasses the distributed architecture and talks to the robot directly.


## Imitation Learning

We train visuomotor policies from demonstrations using two architectures.

### ACT

**Action Chunking with Transformers** <d-cite key="zhao2023act"></d-cite> is a CVAE-based policy. A training-only latent encoder takes demonstrated action chunks and joint positions to infer a latent distribution. Separately, a ResNet processes camera images, and the policy's transformer encoder fuses image features, proprioception, and the latent code. Its transformer decoder predicts a *chunk* of future actions (typically 50-100 timesteps) in one forward pass. At inference, the demonstration-conditioned latent encoder is omitted and the latent is set to the prior mean (zero).

Key properties for our setup:
- ResNet backbone uses pretrained ImageNet weights
- Works well with small datasets (40-100 demonstrations)
- Predicts absolute joint positions
- Training uses L1 reconstruction loss on action chunks plus a weighted KL regularizer on the latent distribution
- With `n_action_steps=50`, simple chunk execution without temporal ensembling means ~5 seconds of open-loop policy commands per prediction at 10 Hz (the servo controller still uses feedback). Reducing this to 5-10 refreshes observations more often, at the cost of more frequent inference.

### SmolVLA

**SmolVLA** <d-cite key="shukor2025smolvla"></d-cite> adapts a vision-language model (VLM) backbone for robot control. The architecture reuses a pretrained SmolVLM-2, whose SigLIP vision encoder and language-model layers process images, instructions, and projected robot state. It generates actions via a **Flow-Matching Transformer action expert**: a learned velocity field is numerically integrated from noise to an action chunk through repeated action-expert evaluations, rather than ACT's CVAE action decoder.

Key properties:
- Pretrained VLM backbone — loading the intended pretrained weights matters. In runs where we accidentally omitted them for months, we saw near-random behavior.
- Rectified flow action head — alternating cross-attention (conditioned on VLM embeddings) and self-attention blocks, trained with a flow-matching objective. The [paper's inference setup uses 10 integration steps](https://arxiv.org/html/2506.01844v1#S4.SS3); the standard implementation uses Euler updates and caches VLM features across the repeated action-expert evaluations. The [official configuration](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/smolvla/configuration_smolvla.py) defaults to `num_steps=10` with caching enabled. This configurable integration-step count is separate from `chunk_size` (actions predicted) and `n_action_steps` (actions executed before refreshing observations).
- Only 450M parameters — designed to run on consumer GPUs or even CPUs.
- Designed for 512x512 image input, though we often train at 128x128 for speed.
- More parameter-efficient fine-tuning via LoRA/PEFT is possible but still under development.
- Pretraining is a plausible source of better data efficiency, but our runs do not isolate its contribution relative to architecture, training settings, or data exposure. We found SmolVLA sensitive to hyperparameters in this setup.

### Training Details

| Setting | ACT | SmolVLA |
|---------|-----|---------|
| Learning rate | 1e-5 | 1e-4 |
| Batch size | 1-64 | 64 |
| Training steps | 10k-100k | 15k-200k |
| Image resolution | dataset native | 128x128-512x512 |
| Pretrained backbone | ResNet (ImageNet) | SmolVLM-2 |
| Action chunk length | 50-100 | 50 |

Findings from our experiments:
- **Batch size and training duration are confounded.** ACT at bs=64 for 10k steps gave about 5/10 cube-sorting successes (the informal, recalled count above), versus the earlier 1/10 at bs=1 for 100k steps. But 64 × 10k = 640k sampled training examples, versus 1 × 100k = 100k, including repeated samples from the same dataset. Fewer optimizer steps do not establish faster wall-clock training or better sample efficiency. These runs do not isolate batch size, exposure, or overfitting as the cause of the difference.
- **Some runs show signs of overfitting.** On our small datasets (40-100 episodes), validation error sometimes rose while training continued, as in the run below. That motivated validation tracking, early stopping, and EMA weight averaging; it does not establish an inevitable 100k-step threshold or prove overfitting is the main cause of rollout failures.
- **Loss is a diagnostic, not rollout success.** ACT's L1/KL objective and SmolVLA's flow-matching objective are not a common action-chunk MSE. Even within an architecture, lower loss is only a proxy. Comparable action-error evaluations require consistent inference settings, preprocessing, normalization, and padding/masking, and must still be checked against robot rollouts.
- **FPV-only is limited by context.** A non-recurrent policy with a single wrist-camera frame and `n_obs_steps=1` has no explicit observation history. We explored 16-frame context (~0.5s at our 30 Hz recording rate), but search remains unresolved. Longer context (perhaps 2-5 seconds) or recurrent architectures such as xLSTM are hypotheses to test; frame stacking also requires support in the policy and data pipeline.

<figure class="post-figure" data-zoomable>
  <img src="/assets/img/robot-rl/il-training-loss-comparison.png" alt="WandB training-loss curves for ACT and SmolVLA on cube sorting; the objectives are not directly comparable." loading="lazy" />
  <figcaption>WandB training-loss curves on cube sorting. Three ACT runs (orange, green, red) at different learning rates and batch sizes plateau around reported loss 1.5-2.0, while SmolVLA (teal) approaches near zero. These reported losses have not been established as commensurate: ACT trains with L1 reconstruction plus KL regularization, while SmolVLA uses a flow-matching objective. The gap does not show that pretrained VLM features caused better learning or rollout performance. All runs use the same 40-episode FPV dataset; comparable action-error evaluation requires the consistent pipeline described above.</figcaption>
</figure>

<figure class="post-figure" data-zoomable>
  <img src="/assets/img/robot-rl/il-validation-loss-overfitting.png" alt="Validation loss (computed with actual inference logic, not teacher-forced) for a SmolVLA training run on cube sorting." loading="lazy" />
  <figcaption>Validation loss (computed with actual inference logic, not teacher-forced) for a SmolVLA training run on cube sorting. Loss decreases until around 60k steps, then starts climbing and becomes noisy past 80k. The vertical line marks the best checkpoint. This motivated adding early stopping with EMA weight averaging to our training scripts. The rising validation error is a run-specific overfitting signal.</figcaption>
</figure>


## Reward Models

For RL, we need a reward signal. We use three approaches, often in combination.

**Learned reward classifier.** A ResNet-10 binary classifier trained on demonstration images. The last N frames of each successful demonstration are labeled `reward=1.0`, everything else `reward=0.0`. These are heuristic terminal-window labels, not ground-truth success annotations for every frame. The classifier produces a success-related score from a single camera image.

Performance is task-dependent. For tasks with clear visual contrasts (matchbox on a bag vs. empty surface) we observed scores around 0.7 on some positive examples; this is not a calibrated 70% confidence in task success. For cluttered scenes with distracting objects, false positive rates increase. We use `pos_weight` to handle class imbalance in the training loss.

**Torque-based rewards (current proxy).** We penalize motor current as a load proxy. High readings can reflect contact (the table, itself, a joint limit), but also supporting the arm against gravity, acceleration, friction, or gripping. We use the following current-based penalty:

$$\text{penalty} = \text{scale} \cdot \log\left(1 + \exp\left(0.5 \cdot \left(\frac{\sum_i I_i^2}{1000} - 10\right)\right)\right)$$

where $I_i$ is the current on motor $i$ and the scale is tunable. The gripper motor can be excluded since gripping *should* produce current.

**Human-in-the-loop rewards.** During live RL training, a human operator presses number keys (0-9) to provide graded rewards, or minus (-) for punishment. This provides an operator-defined learning signal without a visual reward model.


## Reinforcement Learning
### SAC

The core RL algorithm is **Soft Actor-Critic** <d-cite key="haarnoja2018sac"></d-cite>, an off-policy maximum-entropy method. It jointly trains:

- An **actor** (policy network): outputs a squashed Gaussian distribution over delta joint actions
- A **twin critic**: two Q-networks that estimate expected return, taking the minimum to reduce overestimation bias
- A **temperature** $\alpha$: automatically tuned to target a desired entropy level

We use standard SAC with several additions:
- **Delayed policy updates** (TD3-style): in the setup described here, the actor updates every 4 critic updates, intended to improve stability by giving the critic more updates between policy changes
- **Weight normalization**: `minimal_rl.py` projects trainable Linear weights to unit norm after the relevant optimizer steps, excluding the actor and critic output heads. This constrains weight growth; the standalone script differs from the XQC policy described below.
- **Update-to-data (UTD) ratio of 4**: 4 gradient steps per environment step. This is important because environment steps are expensive (real-time on hardware), so we want to extract as much learning as possible from each transition.

<figure class="post-figure" data-zoomable>
  <img src="/assets/img/robot-rl/rl-base-learning-curves.png" alt="SAC on real hardware: position holding with six position channels, including the gripper." loading="lazy" />
  <figcaption>SAC on real hardware: position holding with six position channels, including the gripper. Top panel: reward (negative distance to home position) approaches zero at roughly 800 environment steps in this run. Second: Q-value estimates settle around -150 to -200. Third: the logged current-derived load metric is around 0.5-1.5 in the plot's units, not calibrated torque. Fourth: control loop frequency holds at 6-8 Hz. Bottom: the entropy coefficient alpha falls from 0.20 to 0.12; this alone does not establish that the policy became deterministic. This was our first end-to-end RL run showing the simple learned behavior described below.</figcaption>
</figure>

<figure class="post-figure" data-zoomable>
  <img src="/assets/img/robot-rl/rl-weight-norm-learning-curves.png" alt="Same position holding task with weight normalization and torque penalty added." loading="lazy" />
  <figcaption>Same position-holding task with weight normalization and a current-based penalty added. Now tracking six panels: total reward, position reward component, current-based penalty (near zero), Q-values (around -5 to -8 rather than roughly -200 in the earlier run), the current-derived load metric, and control frequency. Reward is noisier, with periodic dips, and appears to plateau at roughly 1600 environment steps in this run. Low penalty does not establish minimum contact force, and smaller Q magnitudes do not prove better calibration. Alpha declines more slowly, but this alone does not demonstrate useful exploration. Since both the penalty and normalization changed, these runs do not isolate either change's effect.</figcaption>
</figure>

### Action Spaces

**Absolute vs. delta.** Demonstrations are recorded as absolute joint positions, but the RL policy outputs delta actions: small increments relative to the current position. This keeps each commanded increment small.

$$a_{\text{target}} = q_{\text{current}} + \text{tanh}(\pi(s)) \cdot \Delta_{\max}$$

where $\Delta_{\max} = \text{action\_scale\_per\_s} / \text{fps}$ (default: 50/10 = 5.0 position units per step on all six channels, including the gripper). These are the interface's position units, not a universal degree bound.

**Demo conversion.** When seeding the replay buffer from demonstrations, absolute-position transitions are converted to deltas. If any delta exceeds the action scale, the transition is split into N sub-transitions with linearly interpolated proprioception and zero-order-hold images. This keeps the converted actions within the per-step action scale. These intermediate replay transitions are synthetic, not physically observed: the same camera images are held fixed while joint positions are interpolated.

### Controller and Compliance

The robot is fragile, and RL exploration can put dangerous pressure on the servos. To limit commanded motion under load, we moved from a simple position-command interface to a **low-gain PI controller** with current-based action attenuation. These are load-limiting heuristics, not certified safety measures or calibrated force control: position bounds do not bound contact forces. The experiments require physical supervision, which is not a substitute for a validated safety system.

**PI Compliance.** The controller calculates the error between target and current position and applies a low-gain proportional-integral correction, intended to make it less aggressive about reaching the target. Sustained error and integral accumulation can still produce load.

**Current-Aware Action Limiting.** We supplement the controller with two additional mechanisms:
- **Proactive:** Every delta action from the RL policy is clamped to $\pm 5$ position units per step in this configuration, including the gripper channel.
- **Reactive:** Motor currents are read in real time. When a reading exceeds the configured threshold, the commanded action is attenuated. Movements *back* toward the home position are exempted from this attenuation, but can push further into an obstacle rather than unloading the arm, depending on the contact geometry.

These measures run at the environment level and change the transitions seen by the policy: an attenuated command can produce less motion, while the current-based penalty supplies a learning signal.

### Human-in-the-Loop Rewards

During live RL, a human watches the robot and presses keys to provide reward:

| Key | Reward |
|-----|--------|
| 0-9 | 0.0 to 0.9 (graded) |
| - | -0.5 (punishment) |

The human reward is added to any automated signal in use, such as the current penalty or classifier output. This lets us provide feedback without first training a visual reward model; consistency and timing of the operator feedback remain practical concerns.

### XQC

**XQC: Well-conditioned Optimization Accelerates Deep Reinforcement Learning** <d-cite key="palenicek2026xqc"></d-cite> is an actor-critic algorithm built on SAC, with sample-efficiency results reported in the [paper](https://arxiv.org/html/2509.25174v2). Our fork contains an XQC-inspired implementation, not yet fully deployed on the real robot. Published XQC builds on principles from **Cross-Q** <d-cite key="bhatt2024crossq"></d-cite> but, unlike original Cross-Q, **keeps target critics**. The published method and [official implementation](https://github.com/danielpalenicek/xqc) use:
- **Target networks.** XQC retains target critics for bootstrapping; original Cross-Q removes them and uses a joined forward pass for batch-normalization statistics.
- **Batch Normalization (BN).** Applied to the critic input and between linear layers and activations, with a joined current/next-state-action forward pass for BN statistics. This is part of the paper's optimization-conditioning approach.
- **Weight Normalization (WN).** After optimizer steps, the official code projects incoming weight vectors for each output unit to unit norm, including output kernels by default, not only hidden layers. Together with BN and the critic loss, this is intended to stabilize effective updates.
- **Distributional critic (C51-style).** Instead of predicting only a scalar Q-value, the critic outputs a categorical return distribution with **101 atoms**. It uses categorical cross-entropy against a projected Bellman target. Modeling the return distribution is not automatically calibrated epistemic uncertainty, nor does it by itself guarantee faster learning on our robot.

In our fork, `XQCConfig` defaults to **101 atoms** on `[-5, 5]`, with weight normalization enabled. Its `weight_normalize_` in `modeling_xqc.py` applies `F.normalize(..., dim=1)` to every trainable `nn.Linear.weight`, including actor and critic output layers; the learner calls `on_optimizer_step` after critic/actor updates. By contrast, `robot/minimal_rl.py` uses twin scalar critics trained with MSE and skips modules whose names contain `mu`, `log_std`, or `head` during projection. The output-head exclusion belongs to that standalone SAC script, not the fork's XQC policy.

The XQC policy's `select_action` already temporarily puts the actor in evaluation mode to use BN running statistics for single-observation inference, then restores its prior mode. These source-level checks are not a full reproduction or deployment validation.


## Experimental Settings

We've been working bottom-up, starting with the simplest possible RL problems on real hardware before adding complexity.

**Imitation learning task: pick and place.** Grab a cube (or matchbox) and place it at a target position. Trained from 40-100 teleoperated demonstrations using FPV camera input. This is the main benchmark for our IL policies (ACT, SmolVLA).

**RL task 1: Position holding (joints only).** A deliberately simple RL problem. The reward is the negative distance from the current positions to a home position. No camera, just six position channels including the gripper. Early in the run there was lots of motion; later, the policy reduced motion and held position, with reward near zero at approximately 800 environment steps. That is meaningful learning of a simple behavior, not robust manipulation. The servos' own position-holding control also contributes; a zero-delta baseline would help separate that contribution from the learned policy, but no such comparison is reported here. The run exercised the end-to-end SAC, reward, and robot-communication pipeline.

**RL task 2: Torque minimization (joints only, using a current proxy).** Same as above but with a penalty for high motor-current readings. In the illustrated run, reward appeared to plateau around 1600 environment steps. The objective encourages position holding with a lower current-based penalty. Exploration put concerning loads on the servos, which led us to add the low-gain PI controller and reactive action attenuation described above.

**Next RL tasks (planned):**
- Position holding with vision — same reward, but the policy receives camera images instead of (or in addition to) joint positions. A stepping stone to visual RL.
- Vision-based pick and place — combining the IL task with online RL improvement via ResFiT or direct fine-tuning.


## Open Problems

These are the questions we're currently working on or planning to tackle, roughly in priority order.

**Residual fine-tuning (ResFiT).** Freeze a pretrained IL policy, train a small residual correction on top with SAC. This approach keeps the structure of demonstrations while allowing RL to fix systematic errors. The paper reports a base ACT policy improving from 14% to 64% success on a woolly-ball pick-and-place task, using 134 autonomous RL rollouts (~15 minutes of robot execution) on a dexterous humanoid platform with five-fingered hands <d-cite key="ankile2025resfit"></d-cite>.

**RL with visual observations.** All current RL experiments use only proprioception (joint positions). Our planned camera-conditioned policies would add visual encoders to the actor and critic; data augmentation (DrQ-style) and larger datasets are options to evaluate. The IL policies already use vision, so the question is how to transfer that capability to the RL setting.

**Temporal context for FPV.** A single frame from a wrist camera doesn't contain enough information for many tasks. Options include: stacking past frames as transformer context, using recurrent architectures (xLSTM's fixed-size recurrent state may make longer histories computationally manageable, but does not imply unlimited usable memory), or maintaining an explicit spatial memory.

**Hierarchical RL.** Sparse rewards and long horizons make exploration difficult in our setting; hierarchy is one hypothesis to test. We're sketching a two-level architecture: a high-level manager that selects discrete goal codes every K steps, and a low-level worker that executes. In that proposal, the worker would get dense goal-reaching rewards and the manager would handle sparse task rewards. Hindsight relabeling of the manager's goals could provide additional training signal.

**Better reward models.** The current binary classifier is too coarse. Directions include: dense rewards from visual features (color histograms, optical flow), active querying on ambiguous cases, and multi-modal rewards that combine vision with proprioception and motor-current proxies.
