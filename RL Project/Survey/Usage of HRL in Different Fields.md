## Where HRL Shows Up Beyond Games — A Field Guide (2021–2025+)

# Goal
Scan **as many distinct fields as possible** where **Hierarchical RL (HRL)** is used, then mine ideas we can **transfer** to hex‑based strategy HRL (e.g., option libraries, goal‑conditioning, subgoal curricula, multi-scale planning).

---

## Robotics (manipulation, locomotion, mobile)
- **Sequential / long‑horizon manipulation**
  - *HRL + curriculum demos & goal guidance* (2025): [Sun et al., HCDGP](https://www.sciencedirect.com/science/article/abs/pii/S0952197625008668) — sequential robotic manipulation with HRL + curriculum.  
- **Constrained object manipulation (legged platforms)**
  - *Hierarchical RL for quadruped manipulation* (2025): [Azimi et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC11902496/).  
- **LLM/vision‑guided hierarchical control (frontier)**
  - *Hierarchical action models / cognition bridges* (2025): [RoBridge](https://arxiv.org/html/2505.01709v3), [Hierarchical Action Models](https://arxiv.org/html/2502.05485v3).

**Transfer to wargame HRL:** curriculum‑guided skill learning, subgoal generators at multiple temporal scales, safe contact/constraint handling analogues for terrain/supply constraints.

---

## Autonomous Driving (planning, decision‑making)
- **LLM‑augmented HRL for driving** (2025): [Li et al.](https://www.sciencedirect.com/science/article/abs/pii/S0957417425023541).  
- **RL motion‑planning surveys (HRL featured)** (2025): [Z. Li et al. survey](https://arxiv.org/pdf/2503.23650).  
- **Scenario‑based HRL for automated driving** (2025): [SAD‑RL](https://www.researchgate.net/publication/393183392_Scenario-Based_Hierarchical_Reinforcement_Learning_for_Automated_Driving_Decision_Making).

**Transfer:** scenario abstractions ↔ operational “phases,” leader‑follower macro policies, hierarchical safety constraints.

---

## UAVs / Aerial Robotics (inspection, swarms, combat)
- **Viewpoint planning via HRL** (2025): [Wu et al.](https://www.mdpi.com/2504-446X/9/5/352).  
- **Leader–follower HRL for UAV cooperation** (2025): [Zhang et al.](https://arxiv.org/html/2501.13132v1).  
- **Multi‑robot hierarchical safe RL** (2025): [Sun et al.](https://www.nature.com/articles/s41598-025-89285-6).

**Transfer:** hierarchical role assignment (scouts, artillery, armor), safe subgoal selection, multi‑agent coordination primitives.

---

## Traffic Signal Control & Mobility Systems
- **HiLight: HRL with global adversarial guidance for large‑scale TSC** (2025): [HiLight](https://arxiv.org/html/2506.14391v1).  
- **Hierarchical federated RL for adaptive TSC** (2025): [Fu et al.](https://arxiv.org/pdf/2504.05553).  
- **Recent reviews** (2025): [Oxford Academic review](https://academic.oup.com/iti/article/8125227), [MDPI review](https://www.mdpi.com/2412-3811/10/5/114).

**Transfer:** city‑block ↔ region graphs; adversarial/global critics to keep local tactics aligned with global flow.

---

## Building Energy / HVAC Control
- **Year‑round HRL for HVAC** (2025): [Liao et al.](https://www.sciencedirect.com/science/article/abs/pii/S030626192500546X).  
- **Enhanced HRL for chiller plant co‑optimization** (2025): [Zhou et al.](https://www.sciencedirect.com/science/article/abs/pii/S2352710225009003).  
- **ReeM (HRL‑style ensemble thermodynamics)** (2025): [ReeM](https://arxiv.org/html/2505.02439v1).

**Transfer:** hierarchical setpoints ↔ operational subgoals (supply, morale, readiness); comfort/energy tradeoffs ↔ attrition/tempo tradeoffs.

---

## Healthcare (diagnosis dialogues, treatment planning, ops)
- **HRL for automatic disease diagnosis** (2022): [Zhong et al., *Bioinformatics*](https://academic.oup.com/bioinformatics/article/38/16/3995/6625731).  
- **HRL DM for MI‑style counselling dialogues** (2025): [Dialogue Manager](https://arxiv.org/html/2506.19652v1).  
- **Clinical RL reviews** (2025): [Frommeyer et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12295150/), [Banumathi et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12096033/).

**Transfer:** phase‑based dialogues ↔ campaign phases; HRL improves sample‑efficiency in sparse outcomes.

---

## Recommender Systems / E‑commerce
- **HRL‑Rec (AAAI’21)**: channel→item two‑level policy (integrated recommendation): [Xie et al.](https://ojs.aaai.org/index.php/AAAI/article/view/16580) (PDF: [link](https://cdn.aaai.org/ojs/16580/16580-13-20074-1-2-20210518.pdf)).  
- **HRL for novelty‑seeking intent** (2023): [Li et al.](https://arxiv.org/abs/2306.01476).  
- **HRL‑PRP for POI recommenders** (IJCAI’24): [paper](https://www.ijcai.org/proceedings/2024/272).

**Transfer:** high‑level “channel” ↔ operational objective class (defend/attack/resupply); low‑level item selection ↔ tactical option choice.

---

## Finance / Banking
- **Hierarchical Reinforced Trader (bi‑level)** (2024): [HRT](https://arxiv.org/html/2410.14927v1).  
- **BANK‑RL (hierarchical multi‑agent bank management)** (2024): [SSRN](https://papers.ssrn.com/sol3/Delivery.cfm/5071840.pdf?abstractid=5071840&mirid=1).  
- **Survey of RL in Quant Finance** (2025): [ACM CSUR](https://dl.acm.org/doi/10.1145/3733714).

**Transfer:** strategic asset allocation ↔ force disposition; execution policy ↔ tactical maneuver.

---

## Warehousing / Multi‑Robot Logistics
- **Real‑time MRTA with HRL** (2025): [ACM DL version](https://dl.acm.org/doi/pdf/10.5555/3709347.3743793) | [arXiv](https://arxiv.org/html/2502.16079v1).  
- **Adaptive bi‑level task allocation under uncertainty** (2025): [Lin et al.](https://arxiv.org/pdf/2502.10062).

**Transfer:** hierarchical pick‑assign‑route ↔ strategic‑operational‑tactical orders; temporal‑logic constraints ↔ rules of engagement.

---

## Cloud / HPC Scheduling & Resource Management
- **HeraSched: HRL‑based HPC scheduler** (2025): [Springer](https://link.springer.com/article/10.1007/s11227-025-07396-3).  
- **DRL scheduling survey (inc. HRL patterns)** (2025): [arXiv](https://arxiv.org/pdf/2501.01007).  
- **Complex resource networks via HRL + distillation** (2025): [RG preprint](https://www.researchgate.net/publication/393117905_Hierarchical_Reinforcement_Learning_with_Self-Distillation_for_Resource_Scheduling_in_Complex_Resource_Networks).

**Transfer:** job queues ↔ objectives queues; partitioned clusters ↔ theaters/regions.

---

## Wireless / Telecom (spectrum, routing, TN‑NTN)
- **HDRL for TN–NTN spectrum allocation** (2025): [arXiv](https://arxiv.org/html/2501.09212v1).  
- **AI‑enabled routing surveys** (2025): [ScienceDirect survey](https://www.sciencedirect.com/science/article/pii/S111001682500122X).  

**Transfer:** layered networks ↔ layered commands; interference constraints ↔ contested terrain constraints.

---

## Industrial Manufacturing / IoT
- **Hierarchical multi‑policy DRL for IoT manufacturing** (2025): [Knowledge‑Based Systems](https://www.sciencedirect.com/science/article/abs/pii/S0278612525000809).  

**Transfer:** station/line/theater hierarchy ↔ tactical/operational/strategic; WIP flow ↔ supply lines.

---

## Dialogue Systems / NLP Task Planning
- **Hierarchical action abstraction for dialogue policies** (LREC’24): [Cho et al.](https://aclanthology.org/2024.lrec-main.408.pdf).  
- **Open‑domain HRL (utterance/topic levels)** (AAAI’20): [Saleh et al.](https://aaai.org/ojs/index.php/AAAI/article/view/6400).  
- **Survey on RL for dialogue policy** (2023): [Kwan et al.](https://link.springer.com/article/10.1007/s11633-022-1347-y).

**Transfer:** stage controllers ↔ campaign phases; subgoal‑driven utterances ↔ subgoal‑driven maneuvers.

---

## Urban Mobility Beyond Signals (routing, AV + infrastructure)
- **Traffic + AV co‑control via HRL** (2023): [NavTL on ACM DL](https://dl.acm.org/doi/10.1145/3580305.3599839).  
- **Pedestrian‑aware HRL for signals** (2025): [ResearchGate entry](https://www.researchgate.net/publication/388758236_Traffic_signal_optimization_using_hierarchical_reinforcement_learning_incorporating_pedestrian_dynamics_and_flashing_light_mode).

**Transfer:** multi‑agent HRL with shared global objectives; region‑level critics.

---

# Patterns to Reuse in Wargame HRL
- **Two‑level “channel→item” decomposition** (recsys) → *Operational objective class → Tactical option*.  
- **Leader–follower & federated hierarchies** (UAV/TSC) → *Army group ↔ divisions; decentralized local learners + global aggregation*.  
- **Curriculum + goal‑guided HRL** (robotics) → *terrain & objective curricula with subgoal relabeling*.  
- **Region graphs & global critics** (TSC/HVAC) → *strategic oversight to keep tactics aligned with campaign goals*.  
- **Safety & constraint layers** (UAV/healthcare) → *rules of engagement, collateral constraints, supply safety margins*.

---
