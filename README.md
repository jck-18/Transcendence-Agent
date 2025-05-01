# 🧠 Transcendence Agent Gridworld

This is a toy reinforcement learning environment designed to explore a deceptively deep question:

> **What if an agent could learn to stop trying?**

### 🌍 The Environment

A 10x10 Gridworld with:
- Random wall generation per episode
- A goal state (top-right corner)
- A start state (bottom-left corner)
- Some episodes are **unsolvable** — the goal is blocked by walls

The agent doesn't know in advance if the goal is reachable.

### 🤖 The Agent

A simple, greedy policy:
- Moves toward the goal using Manhattan distance
- Tracks its own **frustration**: if it makes no progress for `n` steps, it **opts out**

### ✨ Transcendence Mechanic

- Action 4 is “opt out” — an intentional decision to stop trying
- If the agent opts out **when the goal is unreachable**, it receives **+1 reward**
- If it opts out **when the goal was reachable**, it receives **-1 reward**
- If it **reaches the goal**, it receives **+1 reward**

This tests **meta-cognition**: recognizing futility, not just maximizing reward.

### 📈 Demo Result (Heuristic Agent)

```bash
Episode 1: Reward = 1.0 (goal reached)
Episode 2: Reward = 1.0 (opted out, unreachable)
Episode 3: Reward = -1.0 (opted out early)
```
🧪 Usage
```bash
python train.py
```
To install dependencies:

```bash

pip install -r requirements.txt
```
This project is part of an exploration into modeling transcendence, futility, and intelligent surrender within bounded systems.

No deep learning. Just deliberate failure.

yaml


---

## ✅ 2. `requirements.txt`

```txt
gym
numpy
matplotlib
Run:

bash
Copy
Edit
pip freeze > requirements.txt
To capture any exact versions you're using.

