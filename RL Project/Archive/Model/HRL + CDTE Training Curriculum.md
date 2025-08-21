# HRL + CTDE Training Curriculum

**Goal:** progressively train the hierarchy so that each layer builds on a competent foundation.

---

## Stage 1 — Micro Tactical (π_L with CTDE)

- **Train only the tactical controllers (unit-level).**  
- Use **Centralized Training, Decentralized Execution (CTDE)**:  
  - Each unit has its own local policy (with action masking).  
  - Centralized critic sees the full state to guide training.  
- **Rewards:** simple, dense signals like:
  - Zone control
  - Damage dealt / damage avoided
  - Holding a key tile for N turns
- **Result:**  
  - π_L learns to move, attack, hold positions, and respect supply/fog.  
  - Can handle small maps (e.g. 9×9 skirmishes).

---

## Stage 2 — Operational + Tactical (π_M + π_L)

- **Introduce the operational policy (π_M).**  
  - Outputs maneuver tokens: flank, breakthrough, hold, feint, screen.  
  - Conditions tactical policies on maneuver choice.  
- **Training:**  
  - Reward π_M for maneuver success (e.g., successful flank or encirclement).  
  - Continue updating π_L at a reduced learning rate (so it adapts to maneuvers but doesn’t collapse).  
- **Result:**  
  - Squads and unit groups execute coherent maneuvers.  
  - Tactics align with mid-level intentions.

---

## Stage 3 — Strategic + Operational + Tactical (π_H + π_M + π_L)

- **Add the strategic policy (π_H).**  
  - Chooses subgoals: capture city, control river line, secure supply hub.  
  - Subgoals are passed down as tokens to π_M.  
- **Rewards:**  
  - Sparse, long-horizon signals: VP changes, objective captures, supply integrity.  
  - Use potential-based shaping for intermediate feedback (e.g., distance-to-goal).  
- **Training:**  
  - Freeze or slow π_L updates initially.  
  - Train π_H with actor-critic or imagination (if using world model).  
- **Result:**  
  - Full 3-level hierarchy executes layered goals on 25×25 campaigns.

---

## Why staged training?

- Avoids instability from trying to solve sparse rewards from scratch.  
- Lower layers learn **primitives** first, which higher layers reuse.  
- Mirrors military command structure:  
  - Soldiers (tactics) → Squads (maneuvers) → Generals (strategy).

---

## Practical Tips

- **Freeze/unfreeze:** when introducing a new layer, freeze lower ones for stability.  
- **Replay buffers:** store goal/maneuver tokens so hindsight relabeling accelerates learning.  
- **Evaluation:**  
  - Stage 1: tactical winrate in skirmishes.  
  - Stage 2: maneuver success rate.  
  - Stage 3: VP swing / strategic winrate.
