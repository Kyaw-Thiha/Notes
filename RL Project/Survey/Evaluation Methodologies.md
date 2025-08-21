
# Evaluation Methodology — RL (and HRL) for Hex-Based, Turn-Based Strategy

Goal: a rigorous, reproducible evaluation plan you can use **during training** and **after training** for your stack (MAXQ + Option-Critic + HIRO + Intrinsic Motivation) scaling from **5×5 → 25×25** hex maps with multi-unit control.

---

## A. Core Reporting & Statistics

| Evaluation axis | Metric / protocol | Why it matters for your HRL wargame | How to compute / report | Key papers (links) |
|---|---|---|---|---|
| **Aggregate performance with uncertainty** | Interquartile Mean (IQM), median, mean; **performance profiles**; **APoI** (Average Probability of Improvement) | Avoids cherry-picking; robust to outliers and few-seed regimes | Report IQM ± 95% CI across seeds; show performance profiles over thresholds | Agarwal et al. 2021 (rliable) [abs](https://arxiv.org/abs/2108.13264), [pdf](https://papers.nips.cc/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf) |
| **Reproducibility** | Multiple seeds (≥10 if feasible), bootstrap CIs, exact environment versions & hyperparams | Stabilizes conclusions under stochasticity in long horizons | Fix seeds; log env/game build IDs; report CIs + per-seed curves | Henderson et al. 2018 [abs](https://arxiv.org/abs/1709.06560), [pdf](https://dl.acm.org/doi/pdf/10.5555/3504035.3504427) |
| **Sample & compute efficiency** | AUC (area under learning curve); wall-clock hours; env steps; accelerator-hours | Practicality for large maps; honest compute accounting | Normalize final score vs steps and hours; show time-to-X (e.g., 60% win-rate) | Agarwal et al. 2021 [pdf](https://papers.nips.cc/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf) |

---

## B. Strategy/RTS-Style Outcome Metrics

| Evaluation axis | Metric / protocol | Why it matters | How to compute / report | Key papers (links) |
|---|---|---|---|---|
| **Win-rate vs baselines** | % wins vs heuristic AIs, scripted AIs, past checkpoints | Primary objective; sanity check | Balanced matchups on fixed scenario sets; report CIs | ELF Mini-RTS [pdf](https://papers.neurips.cc/paper/6859-elf-an-extensive-lightweight-and-flexible-research-platform-for-real-time-strategy-games.pdf) |
| **League/Elo/TrueSkill** | Elo/MMR/TrueSkill ratings; ladder performance; league training | Meaningful rankings across many opponents/versions | Round-robin or league; report μ−3σ (conservative skill) | TrueSkill (Herbrich et al. 2006) [pdf](https://papers.neurips.cc/paper/3079-trueskilltm-a-bayesian-skill-rating-system.pdf) |
| **Exploitability / NashConv** | NashConv / exploitability (lower is better) on simplified subgames | Measures robustness to best-response opponents | Use OpenSpiel exploitability tools on abstractions of your game | OpenSpiel (Lanctot et al. 2019) [pdf](https://nessie.ilab.sztaki.hu/~kornai/2022/HaladoGepiTanulas/lanctot_2020.pdf) |
| **Sequential testing for matches** | **SPRT** to accept/reject Elo improvements | Efficiently detects real strength gains | Configure H0/H1 bounds; stop when evidence is sufficient | SPRT overview (chess) [wiki](https://www.chessprogramming.org/Sequential_Probability_Ratio_Test) |

---

## C. Generalization & Robustness (Map Size, Terrain, Opponents)

| Evaluation axis | Metric / protocol | Why it matters | How to compute / report | Key papers (links) |
|---|---|---|---|---|
| **Procedural generalization** | Train on distribution A; test on held-out seeds B (OOD) | Shows transfer from 5×5 to 25×25 & unseen terrains | Use PCG for terrain/supply graphs; report train vs test gaps | Procgen benchmark [pdf](https://proceedings.mlr.press/v119/cobbe20a/cobbe20a.pdf) |
| **Domain randomization** | Randomize terrain costs, weather, fog, supply | Prevents overfitting to a few maps | Stress tests over parameter sweeps | Procgen [abs](https://arxiv.org/abs/1912.01588) |
| **Opponent robustness** | Self-play, **league** with exploiters; PSRO | Avoid brittle strategies; handle non-transitivity | Maintain population; evaluate exploitability trends | PSRO (Lanctot et al. 2017) [pdf](https://mlanctot.info/files/papers/nips17-psro.pdf); AlphaStar league (2019) [pdf](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf) |

---

## D. Multi-Agent (CTDE) & Coordination

| Evaluation axis | Metric / protocol | Why it matters | How to compute / report | Key papers (links) |
|---|---|---|---|---|
| **Credit assignment** | COMA counterfactual advantage; difference-rewards ablations | Quantifies coordination improvements | Compare with/without counterfactual baseline; report team vs individual returns | COMA [abs](https://arxiv.org/abs/1705.08926), [pdf](https://www.cs.ox.ac.uk/people/shimon.whiteson/pubs/foersteraaai18.pdf) |
| **Coordination metrics** | Friendly-fire rate; overkill ratio; synchronous maneuvers; supply-line uptime | Captures emergent coordination beyond win rate | Log event-based KPIs per episode; report distributions | μRTS evaluation templates [abs](https://arxiv.org/abs/2105.13807), [pdf](https://ieee-cog.org/2021/assets/papers/paper_174.pdf) |

---

## E. HRL-Specific Diagnostics (Options, Subgoals, Hierarchies)

| Evaluation axis | Metric / protocol | Why it matters | How to compute / report | Key papers (links) |
|---|---|---|---|---|
| **Option usage & commitment** | Avg option length; termination rate; switching cost | Detects thrashing vs sustained maneuvers | Histogram option durations; add deliberation-cost ablations | Option-Critic [pdf](https://arxiv.org/abs/1609.05140) |
| **Diversity of skills** | Entropy of option selection; MI-based skill diversity (e.g., DIAYN’s $I(S;Z)$) | Ensure a broad skill “toolbox” | Train a skill discriminator; report MI estimates | DIAYN [html](https://ar5iv.labs.arxiv.org/html/1802.06070); DEOC [pdf](https://www.mcgill.ca/science/files/science/channels/attach/diversity_enriched_option_critic.pdf) |
| **Subgoal success** | Subgoal completion rate; success-weighted path efficiency (SPL-style) | Validates hierarchical credit and goal reachability | Success indicator × (shortest-path/actual-path) | SPL background [abs](https://arxiv.org/abs/1807.06757) |
| **Inter-level credit** | Return decomposition across levels; RUDDER/redistribution ablations | Attributes delayed victory to key decisions | Compare with/without redistribution; time-lag sensitivity | (Use as optional ablation alongside HRL diagnostics) |

---

## F. Curriculum-Aware Evaluation (Tactical → Operational → Strategic)

| Evaluation axis | Metric / protocol | Why it matters | How to compute / report | Key papers (links) |
|---|---|---|---|---|
| **Learning Progress (LP)** | **ALP-GMM** score; teacher-student curricula | Ensures difficulty pacing across layers | Track absolute LP; auto-sample tasks with max LP | ALP-GMM [abs](https://proceedings.mlr.press/v100/portelas20a.html), [pdf](https://proceedings.mlr.press/v100/portelas20a/portelas20a.pdf) |
| **Stage-wise transfer** | Zero-shot / few-shot performance after each stage | Verifies tactical→operational→strategic transfer | Freeze lower layers; evaluate higher-layer learning curves | TSCL [abs](https://arxiv.org/abs/1707.00183); Survey [pdf](https://jmlr.org/papers/volume21/20-212/20-212.pdf) |
| **Map-size scaling curves** | Performance vs map size (5×5, 10×10, 15×15, 25×25) | Explicitly tests size generalization | Report win-rate & efficiency across sizes | μRTS/RTS benchmarking patterns [abs](https://arxiv.org/abs/2105.13807) |

---

## G. Interpretable Operational KPIs (Wargame-Specific)

> Use alongside global metrics to diagnose *why* a policy wins/loses.

- **Supply-line disruption time** (turns enemy is out of supply)  
- **Encirclement count & duration** (distinct pockets created × held turns)  
- **Breakthrough depth** (max contiguous advance from front line)  
- **Attrition & exchange ratio** (losses inflicted / losses sustained)  
- **Objective control stability** (variance of VP control over time)

These are domain KPIs you’ll log from engine events and summarize per scenario.

---

## H. Benchmarking & Protocols

- **Scenario suites**: fixed scripted scenarios (opening, river crossing, city siege); **procedurally generated** random maps for generalization.  
- **Match protocols**: best-of-N with balanced sides; mirrored spawns; time control (fixed think-time per turn).  
- **Ablations**: (1) remove PBRS, (2) remove HER, (3) remove deliberation cost, (4) disable intrinsic bonuses, (5) HRL→flat baseline.  
- **Compute disclosure**: CPUs/GPUs/TPUs, hours, frames/steps, replay ratio, batch sizes.  
- **Artifacts**: code, configs, seeds, pre-trained checkpoints, evaluation scripts.

---

## Minimal “What to Report” Checklist

1. **Curves**: IQM ± 95% CI, AUC, time-to-X across seeds.  
2. **Match results**: Elo/TrueSkill ladders, SPRT acceptance.  
3. **Robustness**: train/test split by map seeds & parameters (PCG), opponent pools.  
4. **HRL diagnostics**: option duration/entropy, subgoal success, termination rates.  
5. **Coordination**: COMA vs no-COMA; overkill/FF rates; supply-uptime.  
6. **Scaling**: map-size performance curve; stage-wise transfer table.  
7. **Compute**: wall-clock, steps, accelerator-hours; reproducibility kit.

