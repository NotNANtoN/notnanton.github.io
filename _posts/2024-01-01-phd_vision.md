---
layout: distill
title: My PhD Vision
description: Where I want to go with my research and how I plan to get there
tags: agi cognition
giscus_comments: true
date: 2024-01-01
featured: false

authors:
  - name: Anton Wiehe
    url: "https://notnanton.github.io"
    affiliations:
      name: PHAROS Labs / AdaLab

toc:
  - name: Overview
    subsections:
      - name: Goal
      - name: Strategy
  - name: Cognitive Architecture
    subsections:
      - name: Concept
      - name: Individual Modules
      - name: Learning Hierarchies
  - name: Research Directions

---

## Overview

### Goal
An intelligent agent in direct interaction with its environment, capable of quick in-context learning.

### Strategy
Set up a cognitive architecture with a plan to train and evolve it.
For the PhD: focus on individual aspects and study how they compose in concrete projects.

Concrete plan:
1. Get a suitable testing ground (robot with good simulation, or a decent grid world).
2. Survey existing AGI architectures and plans (Sutton's Alberta Plan, Schmidhuber's Big Net, LeCun's JEPA, etc). Write a paper summarizing the key ingredients and where they diverge.
3. Design my own plan and describe in detail how the modules interact.
4. Work on the most promising module. Current bet: meta-learning with adapters and mixture-of-experts.
5. Select up to 3 modules that can be researched during the PhD and publish them along with the overarching architectural plan.

## Cognitive Architecture

### Concept
The idea is an overall cognitive architecture made of specialized but interconnected modules, loosely inspired by how the brain organizes perception, motivation, and action. Below I sketch the modules and the role each one plays.

### Individual Modules

**Policy network(s).**
A division of labor: multiple policy networks, each responsible for different actuator regions, but with overall connectedness between them. Think of it like the motor cortex, where regions specialize but still coordinate.

**Value network.**
Outputs a multi-faceted reward signal. The analogy here is the brain's control of endorphins and other neurochemicals. Rather than a single scalar reward, the value network produces a richer signal that captures different dimensions of "how are things going."

**Intrinsic motivation network(s).**
Separate from external reward. These networks generate internal rewards for things like curiosity, pain avoidance, beauty-seeking, and self-preservation. The point is that an agent shouldn't need an external teacher to want to explore or to avoid breaking itself.

**Perception network.**
Produces the input for the policy network by building an internal model of the world. It applies sensory biases to filter noise and generates a coherent, multi-modal integration of all senses. This module should be very robust and change slowly, mostly at the beginning of "life" and through evolutionary pressure.

**Affordance network.**
Takes in the current state and an intention, then projects what actions are possible and what their consequences look like. This could be a conditional mode within the perception network, or a standalone module. It answers the question: "given where I am and what I want, what can I do?"

**World model network.**
Strongly connected to the affordance network. Predicts the next state given current state and action. This is what lets the agent plan by simulating futures before committing to action.

### Learning Hierarchies
Learning happens at different levels, each on its own timescale:

1. **Evolutionary.** Defines base reactive behavior, main cognitive biases, and architectural improvements. Implemented as an evolutionary algorithm, guided by epigenetic variations (for example, network node stability and importance influencing how much weights are allowed to change, similar to ideas in the Uber AI paper on protecting learned knowledge).

2. **Lifelong.** Integration of facts, long-term memory, core behaviors, and automatisms. The policy network is trained here, possibly through meta-learning over situational episodes. This is where skills become permanent.

3. **Situational.** Quick adaptation using adapters (like LoRAs). The agent encounters a new situation and rapidly adjusts. If an adapter proves useful enough over time, it gets integrated into the base network, graduating from situational to lifelong knowledge.


## Research Directions

A few directions I find most promising, roughly grouped:

**Incorporating foundation model knowledge into RL agents.**
Can we use what CLIP or LLMs already know and plug it into a reinforcement learning agent? There are several angles: using them as a knowledge base, repurposing their token embeddings as a latent space for "thinking," or treating their learned algorithms as general-purpose tools even in non-language domains.

**Planning as internal simulation.**
Put planning steps (the interaction between a world model and a policy) into recurrent memory so the agent can "think" before acting. It simulates paths, evaluates them, and only commits to action once planning is done.

**Hierarchical RL with learned abstractions.**
Hierarchical actor-critic methods where options and sub-tasks are discovered automatically through quantized, symbol-like representations (in the spirit of VQ-VAE). The goal is an agent that builds its own action hierarchy rather than having one hand-designed.

**Meta-learning for faster adaptation.**
Defining learnability itself through meta-learning: training a world model that learns how to learn, so the agent can pick up new tasks with minimal experience. Related: dividing datasets into clusters, training with MAML across them as separate "skills," and increasing cluster granularity over time.
