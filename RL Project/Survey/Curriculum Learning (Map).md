
# Survey — Exploration & Curriculum Learning in HRL

Target stack context: **MAXQ + Option-Critic + HIRO + Intrinsic Motivation** for scaling from **5×5 → 25×25 hex maps** with varied terrain and multi-unit control.

Exploration and curriculum learning are critical because hierarchy alone does not guarantee transfer from small → large maps. Efficient discovery of subgoals, automatic scaling of difficulty, and robustness to varied terrains/opponents matter directly to your war game AI.

---

## Table: Exploration & Curriculum Learning Approaches

| Technique | What it does (key idea) | Why it’s relevant to your HRL stack | Where to apply (tactical / operational / strategic) | Key papers (links) |
|-----------|--------------------------|-------------------------------------|----------------------------------------------------|-------------------|
| **Curiosity / ICM (Intrinsic Curiosity Module)** | Uses prediction error of forward/inverse dynamics as exploration bonus. | Encourages unit-level scouting in unknown terrain; speeds up discovery of chokepoints/resources when external reward is sparse. | Tactical (unit movement, recon). | Pathak et al. 2017 ([pdf](https://arxiv.org/pdf/1705.05363.pdf)) |
| **Random Network Distillation (RND)** | Predict a fixed random network; prediction error gives novelty bonus. | Simple, scalable exploration signal for large 25×25 maps with fog of war. | Tactical/operational (map exploration, supply disruption). | Burda et al. 2018 ([pdf](https://arxiv.org/pdf/1810.12894.pdf)) |
| **DIAYN / Skill Discovery** | Learn diverse skills by maximizing MI between skill ID and states. | Pretrains maneuver primitives (flanking, probing, holding ground) → useful as options in Option-Critic/HIRO. | Pretraining phase; option discovery. | Eysenbach et al. 2018 ([pdf](https://arxiv.org/pdf/1802.06070.pdf)) |
| **ALP-GMM (Automatic Curriculum)** | Teacher generates tasks of intermediate difficulty by fitting Gaussian mixture to learning progress. | Automatically creates curriculum of maps/scenarios (e.g., terrain density, unit ratios) to pace training. | Environment/task generator for training pipeline. | Portelas et al. 2020 ([pdf](https://arxiv.org/pdf/1910.07224.pdf)) |
| **Teacher-Student Self-Play** | Teacher adversarially proposes tasks; student solves them; difficulty adapts automatically. | Forces generalization across varying opponent strategies / larger maps. | Strategic level training pipeline. | Sukhbaatar et al. 2018 ([pdf](https://arxiv.org/pdf/1710.09831.pdf)) |
| **Domain Randomization (DR)** | Randomize environment parameters (terrain, supply lines, unit stats). | Prevents overfitting to specific maps; ensures transfer to unseen 25×25 terrains. | Training environment design (all levels). | Tobin et al. 2017 ([pdf](https://arxiv.org/pdf/1703.06907.pdf)) |
| **Self-Play Curricula (AlphaStar, AlphaZero)** | Agents train against copies or past selves; naturally adapts difficulty. | Critical for scaling multi-agent strategy (enemy modeling, counter-strategies). | Strategic/operational training. | Vinyals et al. 2019 (AlphaStar [Nature](https://www.nature.com/articles/s41586-019-1724-z)); Silver et al. 2017 (AlphaZero [pdf](https://arxiv.org/pdf/1712.01815.pdf)) |
| **Reverse Curriculum Generation (RCG)** | Start from goal states, gradually expand outward to harder states. | Helps train long-horizon objectives like encirclement or city capture where end reward is very delayed. | Tactical/operational subtasks with sparse rewards. | Florensa et al. 2017 ([pdf](https://arxiv.org/pdf/1707.05300.pdf)) |
| **Progressive Widening / Region Growing** | Expand explored state regions progressively; focus compute on reachable frontier. | Efficient for large combinatorial hex maps; prevents thrashing in vast state space. | Tactical exploration policy. | Florensa et al. 2019 (Stochastic Region Growing [pdf](https://arxiv.org/pdf/1906.01470.pdf)) |
| **Exploration via Surprise / Bayesian Uncertainty** | Use prediction uncertainty or KL divergence as bonus. | Encourages exploring rarely seen tactical configurations (river crossings, encirclement). | Tactical & operational controllers. | Achiam & Sastry 2017 (Surprise-based [pdf](https://arxiv.org/pdf/1703.01732.pdf)) |
| **Hierarchical Curricula (HRL-specific)** | Gradually increase temporal abstraction or option horizon. | Lets HIRO/OC train on small horizons (short maneuvers) first, then scale up to long encirclement operations. | Meta-controller (option horizon scheduling). | Nachum et al. 2019 (HIRO curriculum [pdf](https://arxiv.org/pdf/1810.06721.pdf)) |

---

## Additional Notes

### Exploration methods to combine
- **Curiosity/RND + DIAYN**: Use at tactical unit level for map exploration + skill diversity.  
- **ALP-GMM + Domain Randomization**: Automatic scaling of maps + stochastic terrains.  
- **Self-Play**: To learn robust opponent strategies at strategic layer.  

### Curriculum strategies for hex-WW2 game
- Start with **tiny maps** (5×5, homogeneous terrain).  
- Gradually increase **terrain complexity** (add rivers, forests, supply lines).  
- Scale **unit count and diversity** (infantry only → add tanks, artillery).  
- Introduce **multi-objective missions** (hold VP vs cut supply).  

---

## How this strengthens your project

- **Scalability**: Prevents collapse when scaling from 5×5 → 25×25.  
- **Transferability**: Agents learn generalizable tactics via domain randomization + curricula.  
- **Multi-agent realism**: Self-play + opponent modeling prepares AI for realistic adversaries.  
- **Efficiency**: Curiosity/RND reduce sample complexity in sparse reward campaigns.  

