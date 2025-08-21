# Analysis of Paper
[arXiv:2502.13918](arXiv:2502.13918])

## What the paper does
- **Algorithm:** AlphaZero-style self-play with MCTS + fully-convolutional recurrent net (“Recall” with progressive loss). Recurrent block can be iterated more times at inference to *try* to handle bigger boards.  
- **Representation:** Image-like stacks of channels for both players, terrain, units per stacking level, phases, etc.  
  - Final input channels: **195 + 12(R+1)**, where *R* is number of future reinforcement steps.  
  - Actions: spatial tensors size **(9S+3)×H×W** (S = stacking levels).  
- **Hex convolutions:** Kernels adapted to hex grids (Hexagdly).  
- **Training setup:** Trained on **5×5 boards** (CPU only). Tested on larger maps by increasing recurrent iterations.

## Results & extrapolation problem
- Strong wins on 5×5 scenarios (asymmetric, symmetric, curriculum, randomized).  
- **Extrapolating to larger maps (up to 12×12): performance drops sharply.**  
- More recurrent iterations did **not** recover performance.  
- Likely causes (per authors):  
  1. Knowledge from small boards doesn’t transfer to bigger maps.  
  2. **Tiny training budget** (~1,100 training steps; replay buffer ~6,500 games, far less than AlphaZero scale).

## Why it didn’t scale
1. **State/action tied to board area** → both inputs and action heads scale with H×W.  
2. **No entity/relational abstraction** → conv nets struggle to express unit-to-unit relations across larger boards.  
3. **Perspective inefficiency** → encoding both players separately wastes data/capacity.  
4. **MCTS branching grows** with board size; fixed search budget = noisy targets.  
5. **Tiny self-play budget** → insufficient data for generalization.

---

## What to do instead

### 1) Representation
- Use **entity/graph encoder** (units as nodes; edges = adjacency/LOS/ZOC).  
- **GNN or transformer** with **relative hex/axial encodings**.  
- **Canonicalize perspective** (always from side to move).  
- **Augment with symmetries** (rotations/reflections).  
- Compress stacking into order-invariant summaries.

### 2) Action space
- **Factorize:** (select unit) → (select target hex) → (select action).  
- Use **pointer networks** over legal candidates with **action masking**.  
- For attacks: small k-hot pointers with legality constraints.

### 3) HRL vs MARL
- Start with **Single-agent HRL**: strategic/operational layer → unit/goal, tactical layer → micro.  
- If micro-coordination failures persist: upgrade tactical to **MARL (MAPPO with CTDE)** but enforce 1-move/turn via proposal-select.

### 4) Planning
- Add **AlphaZero-style MCTS** but over **macro options** (move bundles / goal-conditioned rollouts).  
- Restrict expansion to local neighborhoods.

### 5) Training for generalization
- **PCG curriculum:** scale maps 6×6 → 25×25, randomize terrain, chokepoints, VP layouts, weather, force mixes.  
- **Opponent pool self-play:** mix checkpoints + scripted bots.  
- **Reward shaping:** zone control ticks, supply, expected damage, flanks.  
- Stabilizers: advantage normalization, POPART, entropy anneal.

### 6) Scale data pipeline
- Use **Sample Factory / EnvPool** for high-throughput single-node rollouts.  
- Use **RLlib** if distributed/MARL.  
- Train with **orders of magnitude more self-play** than the paper (~6.5k games is far too little).

### 7) If sticking with their recurrent conv + MCTS
- Add **relational layers** (local attention / GNN) between conv stages.  
- **Scale search budget with map size**.  
- Use **progressive widening** + policy-prior-guided expansion.  
- Keep **hex convolutions**, but don’t rely on them alone.

---

## Minimal practical plan
1. **Baseline (2–3 weeks):** SA-HRL + GNN encoder + masked pointer heads; PPO/APPO; PCG 6×6 → 15×15; self-play vs scripted bots.  
2. **Scale (2–3 weeks):** Curriculum to 25×25; opponent pool; POPART; ablations (GNN vs transformer).  
3. **Tactical boost (optional):** Add macro-MCTS or tactical MARL (proposal-select).  
4. **Ablations:** evaluate across map size, terrain, unit mixes, fog-of-war, with/without HRL/planning/MARL.

---
