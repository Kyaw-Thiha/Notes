## Hybrid HRL Architecture for Hex-Based, Turn-Based Strategy AI

## Overview
A practical HRL stack for scaling from **small (5×5)** to **large (25×25)** hex maps with varied terrain and multi-unit control. This hybrid combines:
- **MAXQ** for hierarchical task/value decomposition,
- **Option-Critic** for learning temporally extended **skills/options**,
- **HIRO** for **off-policy, goal-conditioned** efficiency and stability,
- **Intrinsic Motivation (e.g., DIAYN/adInfoHRL/HIDIO)** for exploration under sparse/long-horizon rewards.

---

## Text Diagram (Architecture Sketch)

- **Observation**: map state, units, terrain, supply, FoW; Actions: turn commands
---> (state $s_{t}$)
- **Strategic Command (MaxQ Root)**: 
  Subtasks: SecureRegion, InterdictSupply, CaptureCity
  Reward: 
---> chooses a subtask $\text{g}^S$ (abstract goal/state region)
- **Operational Command (MAXQ mid-level + HIRO high-level)**:
  Goal-conditioned: g^O in state/latent space (HIRO)
  Sets subgoals: “own bridge B”, “hold ridge R (k turns)”
---> issues subgoal $g^0$ → goal-conditioned low-level
- **Tactical Command (Option-Critic Low-Level)**: 
  Options = learned skills (advance, rotate, flank, fortify)
  Intra-option policies $\pi_{o}$(a|s), terminations $\beta_{o}$(s)
  Intrinsic motivation for exploration/skill discovery
---> executes primitive/discrete actions each turn
- Environment Transition + Logging (off-policy replay buffers)

---

## Module Breakdown

**1) MAXQ (Task/Value Decomposition)**
- Define a **task graph** from Strategic → Operational → Tactical.
- Each node/subtask has termination conditions and (pseudo-)reward shaping aligned to campaign goals.
- Provides **credit assignment & structure**: value functions decompose along the hierarchy.

**2) HIRO (Goal-Conditioned Off-Policy HRL)**
- At the Operational level, use **goal-conditioned policies** and off-policy relabeling.
- High-level proposes subgoal vectors (waypoints/regions/latent goals) for the Tactical level.
- Off-policy relabeling + replay makes scaling feasible (sample efficiency on large maps).

**3) Option-Critic (Learned Options/Skills)**
- Tactical layer learns **temporally extended actions**: maneuver, entrench, screen, probe, rotate.
- Learns **intra-option policies** and **termination** end-to-end; no hand-coded skills required.
- Clean interface for the Operational goals to invoke multiple steps of a skill.

**4) Intrinsic Motivation (Skill Discovery & Exploration)**
- Use MI-based or diversity objectives to learn **distinct, reusable skills** and **encourage exploration**.
- Helps overcome **sparse, delayed** outer rewards (campaign outcomes many turns later).

---

## Training Loop (Sketch)

1. **Strategic (MAXQ root)** selects a subtask → provides **Operational** an abstract objective (region/phase).
2. **Operational (HIRO high-level)** samples a **goal** $g^O$ in state/latent space to achieve within $k$ steps.
3. **Tactical (Option-Critic)** executes options (skills) to move toward $g^O$, with intrinsic bonuses for discovery/diversity.
4. Store transitions in **multi-level replay buffers** (high-level & low-level, plus intrinsic).
5. Optimize:
   - **MAXQ**: update decomposed value functions and completion values per subtask.
   - **HIRO**: off-policy updates with **goal relabeling**; stabilize high-level goals.
   - **Option-Critic**: policy/termination updates for skills; entropy/regularization as needed.
   - **Intrinsic**: maximize skill diversity or predictability (e.g., MI-based objectives).
6. **Curriculum**: expand from 5×5 → 8×8 → 12×12 → 25×25, vary terrain/supply/fog.

---

## Why These Four Are Important (Succinct)

- **MAXQ** → Gives you a **command hierarchy** and **value decomposition**. Essential for clarity, credit assignment, and matching human-like planning layers (strategic/operational/tactical).
- **HIRO** → **Sample efficiency & stability** for large maps by using **off-policy** training and **goal relabeling**. It turns high-level planning into feasible learning.
- **Option-Critic** → Learns **skills/options** automatically (no manual scripting) so the agent can act at **longer temporal scales** (multi-turn maneuvers).
- **Intrinsic Motivation** → Addresses **sparse/delayed rewards** and **exploration** in big, partially observed maps (find bridges, ridges, supply lines) before extrinsic signals arrive.

---

## When to Consider Other Paradigms

- **FeUdal Networks (FuN)**  
  Use when a **strict 2-level** manager–worker structure is sufficient and you want **goal directions** in latent space without a full task/value decomposition tree. Good for simpler, continuous-progress tasks.

- **HAC (Hierarchical Actor–Critic)**  
  Prefer in **continuous-control** settings or robotics-like tasks where **hindsight relabeling** of subgoals across multiple levels helps with sparse rewards and smooth action spaces.

- **Goal-Conditioned RL (beyond HIRO)**  
  If you have **clear, geometric goals** (e.g., reach a hex/region/feature) and want to simplify hierarchy, a strong **GCRL** baseline (HER, TDMs) can work at the operational/tactical split.

- **LLM-Guided HRL / Planning**  
  Consider if you need **rapid strategy prototyping** or **curriculum generation**. Useful for generating subgoal candidates and **explanations**, but adds complexity and may be hard to benchmark rigorously for a first paper.

- **Pure Options/Option Discovery (no MAXQ)**  
  If you prefer **flatter control** with reusable skills and **limited hierarchy**, focus on **Option-Critic + Intrinsic**; simpler to implement but weaker for campaign-level credit assignment.

---

## Practical Notes (Implementation)

- **State Abstractions**: For HIRO, define subgoal spaces (e.g., hex coordinates, region embeddings, or GNN latents). For MAXQ, define termination conditions per subtask (e.g., “own bridge B for ≥k turns”).
- **Skill Library Warmup**: Pretrain Option-Critic with intrinsic rewards (no extrinsic) to bootstrap maneuvers; then introduce extrinsic rewards and curriculum scaling.
- **Logging**: Track per-level returns, subgoal success rates, option usage statistics, MI/diversity scores.
- **Evaluation**: Fixed-seed **unseen maps** (generalization), **terrain shifts**, **supply/fog variations**. Metrics: win rate, objective completion time, attrition ratio, OODA tempo (turns to key subgoals).

---

## TL;DR Recipe
- Use **MAXQ** to formalize the command hierarchy.
- Use **HIRO** at the **Operational** level for off-policy goal-setting.
- Use **Option-Critic** at the **Tactical** level to learn multi-turn maneuvers.
- Add **Intrinsic Motivation** to handle sparse rewards & encourage skill diversity.
- Layer in **curriculum** (map size/terrain variability) for scalable generalization.

