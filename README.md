# mrCPP — Multi-Robot Coverage Path Planning with Dynamic Obstacles

[![JAX](https://img.shields.io/badge/JAX-Enabled-orange?style=flat-square&logo=google)](https://github.com/google/jax)
[![CUDA](https://img.shields.io/badge/CUDA-Accelerated-green?style=flat-square&logo=nvidia)](https://developer.nvidia.com/cuda-zone)

**mrCPP** addresses **multi-robot coverage path planning (CPP)**: a team of robots must jointly visit every reachable cell of an unknown or partially known environment as efficiently as possible, while avoiding collisions with static structures and **dynamic obstacles** (e.g., pedestrians moving through the workspace).

The project explores and compares modern machine-learning approaches to this control problem—ranging from standard model-free MARL to advanced latent world-model methods—to identify optimal architectures for decentralized, scalable, obstacle-aware coverage.

---

## 🧭 Task Definition & Decentralized Execution

Each robot operates under strict **decentralized execution**, relying exclusively on local, egocentric observations:
* **2D LiDAR Scan:** Local range readings capturing static walls and dynamic humans.
* **Ego Velocity:** Linear and angular velocity of the individual robot.
* **Local Coverage Sub-Grid:** A local window of the global coverage map marking visited vs. unvisited cells.
* **Local Teammate Matrix:** Relative state (position/velocity) of nearby teammates within sensing range.

### Environment & Dynamics
* **Unknown Floor Plans:** Episodes take place in randomly generated 2D indoor layouts unknown to the robots at initialization.
* **Dynamic Obstacles:** Simulated using the **Headed Social Force Model (HSFM)**, producing realistic, reactive pedestrian trajectories (goal attraction, inter-agent and wall repulsion) rather than scripted paths.

### Centralized Training, Decentralized Execution (CTDE)
During training, the central critic leverages privileged global state (full robot states, pedestrian trajectories, and the complete environment coverage grid) to provide accurate credit assignment without requiring global observations at test time.

---

## 🧪 Candidate Methods

The project evaluates architectures in increasing order of complexity:

1. **MAPPO with CTDE:** Standard model-free multi-agent PPO baseline with decentralized actors and a centralized critic.
2. **World-Model Methods (Dreamer-style):** Learning a recurrent latent dynamics model per agent to train actor/critic via imagined rollouts in latent space.
3. **Mamba-JEPA (Candidate Best Solution):** A purely latent (non-generative) world model built on a **Mamba** structured state-space model (SSM):
   - **JEPA-style Encoder:** Compresses LiDAR and local sub-grid observations into a latent state without pixel-space reconstruction.
   - **Predictive World Model:** Trained exclusively to minimize future latent state prediction error.
   - **Mamba Core:** Replaces traditional RSSM recurrence to efficiently maintain long historical context and build a shared latent "mental map" across robots.

---

## ⚙️ Implementation Constraints

* **Framework:** 100% implemented in **JAX**, optimized for high-performance **CUDA** execution.
* **Backend selection:** `train_simple.py` picks its device automatically — Apple **Metal** (via `jax-metal`) on Apple Silicon, otherwise **CUDA** when an NVIDIA GPU is visible, otherwise **CPU**. Override with `--backend {auto,metal,cuda,cpu}`.

> **Note on Apple Silicon:** `jax-metal` is experimental. On an M1 Pro the Metal
> backend measures ~2× *slower* than the CPU backend for the baseline config,
> because the networks are small and the rollout is dominated by many tiny
> kernel dispatches. Prefer `--backend cpu` on Mac unless you scale the model up.

* **Monitoring:** training streams to **Weights & Biases** (configured in the
  `wandb:` block of the YAML, overridable with `--wandb-project`, `--wandb-name`,
  `--wandb-mode {online,offline,disabled}`, or switched off with `--no-wandb`).
  Run `wandb login` once first. The same metrics are always written to
  `<save-dir>/training_log.csv`, and a failed W&B init never stops training.

  Logged per `log_interval`: episode reward / coverage / length, the end-cause
  rates (`rate/completion`, `rate/timeout`, `rate/collision_end`), collision
  diagnostics split by cause (`collision/wall_*`, `collision/robot_*`, both as a
  per-robot-step rate and as a per-episode count), the PPO losses, policy sigma
  and the decayed learning rates.

---

## 🚀 Getting Started

Clone the repository and install dependencies:

```bash
git clone [https://github.com/otr-ebla/mrCPP.git](https://github.com/otr-ebla/mrCPP.git)
cd mrCPP
pip install -r requirements.txt

---

