# Bottom-Up Curriculum Learning in Hybrid HRL

## Why Bottom-Up Works
- **Skill library first:** Tactical options (advance, rotate, flank, entrench, probe) are reusable across map sizes. Learning them on **5×5** with dense/shaped rewards is faster.
- **Stable interfaces upward:** Once low-level skills have consistent semantics and termination, Operational (HIRO) can treat them like reliable actuators; Strategic/MAXQ gets cleaner credit assignment.

---

## Practical Recipe (Staged Curriculum)

### Stage A — Tactical-Only Pretraining (5×5 → 8×8)
- **Train:** Option-Critic (or skill discovery + OC) with intrinsic + shaped extrinsic rewards.
- **Goals:** Local maneuvers: “reach hex (x,y)”, “hold tile k turns”, “cross river”, “take ridge”.
- **Outputs:** A **skill library** (options) with intra-option policies π_o and terminations β_o.
- **Keep:** Checkpoints + success detectors for each option; export option interface (name/id, expected horizon, preconditions).

**Tips**
- Use dense local rewards early (distance-to-goal, formation integrity), anneal toward sparser shaping.
- Enforce **option horizon** (e.g., 3–8 turns) so higher layers can reason about time.

---

### Stage B — Freeze-Then-Finetune on Mid Maps (12×12)
- **Plug into HIRO:** Treat each option as part of the tactical action space; Operational outputs subgoals consistent with what options can achieve.
- **Training:** Start with **frozen** tactical weights for stability; unfreeze **termination β** first, then intra-option π if needed.
- **Relabeling:** Use HIRO goal-relabeling with *achieved subgoals* computed from tactical trajectories.

**Tips**
- Add a **handover loss**: penalize operational subgoals that consistently force “bad” options (e.g., crossing impassable terrain).
- Log **option usage entropy**—if one option dominates, your library is too narrow.

---

### Stage C — Integrate MAXQ (Operational/Strategic Subtasks)
- **Define subtasks:** SecureRegion, InterdictSupply, CaptureCity with crisp termination.
- **Curriculum:** Grow map size & variability (terrain, weather, supply, fog) after the agent hits success-rate targets.

**Tips**
- Use **pseudo-rewards** aligned with subtask completion (e.g., “hold bridge ≥ k turns”).

---

## Training Control Knobs
- **Freeze schedule:** (i) freeze OC entirely → (ii) unfreeze β (termination) → (iii) selective π_o finetune.
- **Option pruning/merging:** Remove rarely used or redundant skills.
- **Skill distillation:** Merge similar skills into compact networks.
- **Goal-space alignment:** Ensure HIRO’s subgoal representation is *reachable* by tactical options.

---

## Interfaces to Define Early
- **Preconditions:** terrain, unit state, supply needed to start an option.
- **Expected horizon:** nominal duration in turns.
- **Success detector:** clear signal to β and higher levels (option achieved its micro-goal).
- **Fallbacks:** if option stalls, return control upward with a failure code.

---

## Common Pitfalls
- **Distribution shift:** Options trained on tidy 5×5 maps may fail on chaotic 25×25.  
  → Use **domain randomization** during pretraining (terrain noise, blockers, fog).  
- **Over-specialized skills:** River-only crossing fails elsewhere.  
  → Train with **parameterized options** (e.g., “cross-obstacle(type=river/forest)”).
- **Goal mismatch:** HIRO requests subgoals options can’t achieve.  
  → Add **feasibility classifier** or regularize high-level goal selection.
- **Credit leakage:** Tactical shaping dominates campaign reward.  
  → Anneal shaping; keep MAXQ pseudo-rewards sparse.

---

## Minimal 12-Week Schedule
1. **Weeks 1–3:** Tactical OC on 5×5 with dense local curricula + intrinsic (discover 6–10 options).  
2. **Weeks 4–6:** Freeze OC; train HIRO on 8×8 using those options.  
3. **Weeks 7–9:** Unfreeze β and π_o selectively; introduce terrain variability.  
4. **Weeks 10–12:** Add MAXQ subtasks; scale to 12×12 and 25×25; anneal shaping.

---

## Ablations
- **No tactical pretrain** vs **with tactical pretrain**.  
- **Frozen vs finetuned** tactical layers.  
- **With vs without intrinsic** during pretraining.  
- **Option library size** (small/large) vs performance & compute.

---

## TL;DR
- Yes — **bottom-up curriculum** (tactical → operational → strategic) is not only feasible; it’s recommended.  
- Pretraining on small maps builds a **skill library**; HIRO and MAXQ then learn **how/when** to deploy those skills on larger, varied maps—faster and more stably than end-to-end from scratch.  
- To ensure **tactical layer extrapolates to larger maps & varied terrain**, **Graph Neural Networks (GNNs)** are a strong choice for the policy/state encoder:  
  - They capture **relational inductive bias** (units, hexes, terrain as nodes/edges).  
  - Naturally handle **variable map sizes**.  
  - Support **generalization** across 5×5 → 25×25 maps by reusing graph structure instead of fixed grids.  

---
