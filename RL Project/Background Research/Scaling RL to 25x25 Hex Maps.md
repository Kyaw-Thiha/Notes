# Scaling RL to 25×25 Hex Maps — Key Concepts and Analysis

This page [[Analysis on Closest Paper]] expands on the analysis of , which trained RL agents on 5×5 hex maps but failed to generalize to larger boards. Below are the important ideas, the problems they faced, and the technical components you should know. 

---

## 📝 Summary of the Paper
- Used **AlphaZero-style self-play + MCTS**.
- Model = **fully-convolutional recurrent net** (“Recall”).
- Input = image-like tensor of board state, unit stacks, terrain, phases.
- Actions = tensor sized with map area **(H×W)** → grows with board size.
- Trained on **5×5 maps**, then tested on larger (up to 12×12).
- Result: **failed to extrapolate**; more recurrent iterations did not fix it.
- Likely reasons:  
  - Model tied to grid size.  
  - No relational abstractions.  
  - Limited training budget (~1,100 steps, ~6.5k games).  

---

## ⚠️ Why it Didn’t Scale
1. **State/action grows with map size** → model must relearn on bigger boards.  
2. **No entity/relational reasoning** → can’t capture “unit A flanks unit B.”  
3. **Perspective inefficiency** → both players encoded separately, no symmetry reuse.  
4. **MCTS branching grows with board** → search budget too small.  
5. **Tiny self-play dataset** → far too little to generalize.

---

## ✅ Fixes for Scaling
- **Representation:**  
  - Use **GNN encoder** or **transformer with relative hex embeddings**.  
  - Canonicalize perspective (always from side-to-move).  
  - Apply symmetry augmentations (rotations/reflections).  
- **Action space:**  
  - Factorize → (unit) → (hex) → (action).  
  - Use **masked pointer heads** to only consider legal moves.  
- **Training regime:**  
  - **PCG curriculum**: gradually increase map size with procedurally generated terrains.  
  - **Opponent pool**: self-play against past checkpoints + scripted bots.  
  - Reward shaping: zone control, supply, expected damage, flanking bonuses.  
  - **POPART** for stable value normalization.  
- **Optional improvements:**  
  - **Tactical MARL (MAPPO/CTDE)** if micro-coordination is weak.  
  - **Planning (MCTS over macro-actions)** to boost tactical accuracy.  
  - **Imitation warm-start** with scripted strategies.

---

## 🔑 Important Technical Terms

### 1. GNN Encoder
- **GNN = Graph Neural Network**.  
- Treats the map as a **graph**:  
  - Nodes = units or tiles.  
  - Edges = adjacency, line-of-sight, supply chain, zone-of-control.  
- Passes messages between nodes so units “know about neighbors.”  
- **Benefit:** generalizes across different map sizes and variable unit counts.

---

### 2. Masked Pointer Heads
- **Pointer network head:** an attention-like mechanism that selects one item from a set (e.g., “which unit?” or “which hex?”).  
- **Masking:** illegal actions (blocked terrain, non-movable units) are filtered out so the network doesn’t waste probability on them.  
- **Benefit:** scales to large boards because the agent only chooses among *legal actions*, not the entire map.

---

### 3. PCG (Procedural Content Generation)
- Automatically generate **maps and scenarios** instead of handcrafting.  
- Example: randomize terrain, unit placement, objectives, weather.  
- **Curriculum:** start small (6×6), gradually scale to 25×25.  
- **Benefit:** prevents overfitting to tiny maps, teaches the agent general strategy.

---

### 4. Ablation
- An **ablation study** = remove or modify one system component and measure performance.  
- Example:  
  - With vs. without GNN encoder.  
  - With vs. without masking.  
- **Benefit:** proves which parts of your architecture are essential.

---

### 5. POPART
- **POPART = Preserving Outputs Precisely, while Adaptively Rescaling Targets.**  
- A technique to normalize **value/reward scales** dynamically.  
- Important for long games where rewards vary (small skirmishes vs. final victory).  
- **Benefit:** stabilizes critic/value learning and prevents collapse.

---

### 6. CTDE (Centralized Training with Decentralized Execution)
- A MARL approach where:  
  - Training: critics see the *whole map*.  
  - Execution: each unit acts only with its *local observations*.  
- **Benefit:** units learn to coordinate while remaining decentralized at test time.

---

### 7. Opponent Pool (Self-Play)
- Train against a **mixture of past versions** of the agent + scripted bots.  
- Prevents “forgetting” and helps converge to robust strategies.  
- Often managed via Elo/Nash sampling.

---

### 8. Reward Shaping
- Adding **dense intermediate signals** (zone control, supply protection, flanking) instead of only final win/loss.  
- **Benefit:** improves sample efficiency, avoids sparse reward problem.

---

## 🚀 Minimal Practical Plan
1. **Baseline:** SA-HRL + GNN encoder + masked pointer heads; train on 6×6 PCG maps with PPO/APPO.  
2. **Scaling:** Curriculum to 25×25; add opponent pool, POPART.  
3. **Boost:** Add tactical MARL or macro-MCTS if needed.  
4. **Ablations:** test with/without GNN, masking, HRL to prove necessity.

---
