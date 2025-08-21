
# Survey — Exploration & Curriculum Learning in HRL  
### With focus on **layered curricula** (tactical → operational → strategic)

You are not only scaling **map size (5×5 → 25×25)** but also **hierarchical layer complexity (tactical → operational → strategic)**.  
This requires surveying papers that **progressively unlock abstraction levels** in HRL or use **layered curricula**.

---

## Table: Exploration & Curriculum Learning (Extended with Hierarchical Curricula)

| Technique | What it does (key idea) | Why it’s relevant to your HRL stack | Where to apply (tactical / operational / strategic) | Key papers (links) |
|-----------|--------------------------|-------------------------------------|----------------------------------------------------|-------------------|
| **Layered Curricula in HRL (Progressive Abstraction)** | Train low-level skills first (navigation, locomotion), then train higher-level controllers on top of pretrained options. | Matches your plan: tactical (unit maneuvers) → operational (encirclement/supply) → strategic (campaign planning). | Start with MAXQ/OC at tactical, then unfreeze HIRO/meta-controller. | Nachum et al. 2019 (Meta-HRL pretraining) ([pdf](https://arxiv.org/pdf/1810.06721.pdf)); Heess et al. 2016 (Hierarchical locomotion curriculum) ([pdf](https://arxiv.org/pdf/1611.01796.pdf)) |
| **Option Pretraining → Strategic Curriculum** | First learn reusable option library, then compose them at higher layers. | Your OC layer can discover maneuvers (flank, hold, probe), which HIRO uses at operational/strategic level. | Tactical → Operational/Strategic transfer. | Bacon et al. 2017 (OC + skill pretraining) ([pdf](https://arxiv.org/pdf/1609.05140.pdf)); Sharma et al. 2017 (Option transfer) ([pdf](https://arxiv.org/pdf/1702.03041.pdf)) |
| **Hierarchical Curriculum with Meta-Controller Scheduling** | Curriculum not only over tasks but also over *abstraction horizon*: start with short option horizons, gradually extend. | Prevents high-level controller from thrashing when low-level isn’t mature. | Tactical (short horizon options) → Operational (mid horizon) → Strategic (long horizon). | Nachum et al. 2018 (HIRO) ([pdf](https://arxiv.org/pdf/1805.08296.pdf)); Vezhnevets et al. 2017 (FeUdal nets curriculum) ([pdf](https://arxiv.org/pdf/1703.01161.pdf)) |
| **Reverse Curriculum Generation (RCG) for HRL** | Train low-level controllers backward from goal, then layer higher-level policies that use those skills. | Useful for strategic objectives like “win campaign” by starting from easier tactical subgoals. | Tactical subgoals → operational composites. | Florensa et al. 2017 ([pdf](https://arxiv.org/pdf/1707.05300.pdf)) |
| **Curriculum over Subgoals (Goal-conditioned HRL)** | Progressively train high-level to propose increasingly abstract goals. | Strategic layer initially constrained (e.g., “take nearby VP”), then broaden to long-range multi-turn objectives. | HIRO + HER pipeline. | Levy et al. 2019 (Hierarchical goal-conditioned curriculum) ([pdf](https://arxiv.org/pdf/1906.05862.pdf)) |
| **Multi-Agent Curricula in HRL** | Scale from single-agent tactical skills → multi-agent coordination at higher levels. | Exactly your case: individual units learn tactics, then coordinated operational maneuvers, then strategic campaign. | All levels under CTDE. | Gupta et al. 2017 (MAHRL, “cooperative multi-agent HRL”) ([pdf](https://arxiv.org/pdf/1703.06182.pdf)); Tang et al. 2021 (Multi-agent HRL curriculum) ([pdf](https://arxiv.org/pdf/2102.12646.pdf)) |
| **Hierarchical Task Networks (HTN) + HRL Curricula** | Combine symbolic task decomposition (HTN planners) with HRL, gradually increasing task hierarchy depth. | Relevant for wargame: can encode tactical tasks first, then strategic doctrines. | Tactical HTN → operational/strategic HTN. | Nau et al. 2003 (HTN planning survey) ([pdf](https://www.cs.umd.edu/projects/shop/at/papers/htn-tr.pdf)); Xu et al. 2018 (HTN+HRL) ([pdf](https://arxiv.org/pdf/1803.08294.pdf)) |

---

## Layered Curriculum Design (for Hex-WW2 game)

1. **Stage 1 — Tactical curriculum**  
   - Train on **small maps (5×5, homogeneous terrain)**.  
   - Learn tactical skills (move, attack, hold, probe, retreat).  
   - Intrinsic motivation (ICM, RND) helps exploration.  
   - MAXQ pseudo-rewards or OC skills define clean subtasks.

2. **Stage 2 — Operational curriculum**  
   - Increase map size (10×10, terrain variation).  
   - Compose tactical skills into operational goals (encirclement, supply interdiction).  
   - HIRO relabels subgoals; HER supports credit assignment.  
   - Domain randomization introduces supply variations.

3. **Stage 3 — Strategic curriculum**  
   - Full 25×25 maps with multiple fronts.  
   - High-level controller sets campaign goals (capture, hold, breakthrough).  
   - Self-play and opponent modeling provide adaptive curricula.  
   - Reward machines or LTL shape multi-phase objectives.  

---

## Why this matters

- **Matches real military command structure**: bottom-up training mirrors historical tactical → operational → strategic learning.  
- **Sample efficiency**: ensures each layer learns at its natural scale before burdening higher levels.  
- **Transferability**: options/subgoals discovered at tactical stage become **reusable building blocks** at operational/strategic stages.  
- **Novelty**: Few works explicitly train HRL with *layered curricula across abstraction levels* — your survey will highlight this gap and position your work as filling it.  

