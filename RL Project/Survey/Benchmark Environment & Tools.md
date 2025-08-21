
# Benchmark Environments & Tooling — Hex-Based, Turn-Based Strategy RL

Target: evaluate and iterate an HRL stack (**MAXQ + Option-Critic + HIRO + Intrinsic Motivation**) that scales from **5×5 → 25×25** hex maps with multi-unit control.

---

## A) Turn-Based Strategy (TBS) / Strategy-Game Benchmarks

| Environment / Framework | Type | What you get (mechanics, observability, APIs) | Why it’s relevant to hex-TBS HRL | Link |
|---|---|---|---|---|
| **Stratega** | General **strategy-game** framework (turn-based + real-time) | YAML-defined tiles/units/actions; forward model for rollouts; logging; designed for **statistical forward planning** | Rapid prototyping of TBS tasks with fog-of-war, tech/economy; forward model helps **option evaluation** and HRL+planning hybrids | Paper: https://arxiv.org/abs/2009.05643 • PDF: https://www.diego-perez.net/papers/StrategaDesign.pdf |
| **Tribes** | **Turn-based strategy** game for AI research | Multi-player, partial observability, resource mgmt; open-source; designed for research | Built-in **TBS** dynamics (economy + combat) ideal for curriculum from tactical→operational→strategic | Paper (AIIDE-20): https://cdn.aaai.org/ojs/7438/7438-52-10764-1-2-20200923.pdf |
| **TAG / PyTAG** | **Tabletop/board** game suite + Python interface | Many multi-agent, **turn-based** games; logging of action spaces/branching factor; PyTAG gives a MARL-friendly API | Fast iterations on **turn-taking** credit assignment; good for HRL option discovery & evaluation before moving into full wargames | TAG: https://arxiv.org/abs/2009.12065 • PyTAG: https://arxiv.org/abs/2405.18123 |
| **OpenSpiel** | General **game RL** framework | Dozens of turn-taking games; tools for **exploitability/NashConv**, replicator dynamics; planning + RL baselines | Standard **evaluation tooling** for turn-based multi-agent settings; plug for opponent-robustness tests | Paper: https://arxiv.org/abs/1908.09453 • Code: https://github.com/google-deepmind/open_spiel |
| **Ludii** | General **(board) game system** | High-level ludemes; efficient turn-based simulation; supports **hex** and square boards; thousands of games | Quick prototyping of **hex-grid** mechanics; can test HRL option diversity & transfer across hex variants | Paper (ECAI’20): https://ludii.games/publications/ECAI2020.pdf |
| **Hexboard** | **Hex-grid** TBS framework (GVGAI-inspired) | Generic hex-board games; level generators; research scaffolding | Ready-made **hex** topology for state/action design and curriculum | Paper: https://www.researchgate.net/publication/332075372_Hexboard_A_generic_game_framework_for_turn-based_strategy_games |
| **CivRealm (Freeciv)** | **Civilization-like** environment (turn-based) | Gym-style API to Freeciv-web; supports tensor agents & LLM agents; complex economy/tech | Large-scale **turn-based** sandbox to stress-test transfer from small maps to **full campaigns** | Paper page: https://openreview.net/forum?id=UBVNwD3hPN • Code: https://github.com/bigai-ai/civrealm |
| **Battle for Wesnoth (Gym wrapper)** | Open-source **hex-grid** TBS game + RL env | Community game; **hex tiles**, factions, fog-of-war; 3rd-party Gym-like wrappers exist | A true **hex-and-counter** platform; realistic tactical subtasks (ZOC, terrain, day/night) for HRL | Wrapper example: https://github.com/DStelter94/ARLinBfW • Game: https://wiki.wesnoth.org |

> Also useful (not turn-based but instructive for strategy agents and tooling): **ELF Mini-RTS** (NeurIPS’17) and **Gym-μRTS** (CoG’21). They provide full-game RTS combat, ablation templates, and fast simulation for option-level pretraining before moving to turn-based scenarios.
- ELF Mini-RTS: https://arxiv.org/abs/1707.01067 • PDF: https://papers.neurips.cc/paper/6859-elf-an-extensive-lightweight-and-flexible-research-platform-for-real-time-strategy-games.pdf  
- Gym-μRTS: https://arxiv.org/abs/2105.13807

---

## B) Multi-Agent APIs & Turn-Taking Interfaces

| Tooling | What it is | Why it fits hex-TBS HRL | Link |
|---|---|---|---|
| **PettingZoo (AEC & Parallel APIs)** | **Standard MARL API** with explicit **turn-taking** (AEC) and parallel modes | Clean interface for **CTDE** training and decentralized execution; AEC mirrors strict turn order in hex wargames | Paper: https://arxiv.org/abs/2009.14471 • NeurIPS’21: https://papers.neurips.cc/paper/2021/file/7ed2d3454c5eea71148b11d0c25104ff-Paper.pdf • Docs: https://pettingzoo.farama.org/ |
| **OpenSpiel** | Games + **evaluation** (NashConv, exploitability) | Drop-in **robustness** metrics for your agents, even if you train elsewhere | https://arxiv.org/abs/1908.09453 |

---

## C) Map/Task Generation & Engine-Side Tools

| Tooling | What it is | Why it helps | Link |
|---|---|---|---|
| **Griddly / GriddlyJS** | Configurable grid-world engine + **web IDE** | Rapid prototyping of PCG maps, observation variants, and **logging**; good for ablation of observation/state encodings | Engine: https://arxiv.org/abs/2011.06363 • Web IDE (Datasets & Benchmarks): https://proceedings.neurips.cc/paper_files/paper/2022/file/611b896d447df43c898062358df4c114-Paper-Datasets_and_Benchmarks.pdf |
| **Red Blob Games — Hex Grids** | Definitive engineering guide for **hex** coords, range, pathing | Saves weeks on hex-math; ensures consistent state/action encoding for HRL | https://www.redblobgames.com/grids/hexagons/ |
| **SGDL (Strategy Game Description Language)** | Language for describing strategy games (rules, maps, units) | If you need a **DSL** to systematically vary rules/maps for curricula and transfer | Survey/paper: https://julian.togelius.com/Mahlmann2011Modelling.pdf |

---

## D) Throughput, Orchestration, and Baseline Stacks

| Tooling | What it is | Why it helps this project | Link |
|---|---|---|---|
| **EnvPool** | High-throughput **vectorized env** execution (C++/pybind11) | Turn-based games can still be throughput-bound; EnvPool gives **×(10–100)** speedups for training and evaluation sweeps | Paper: https://arxiv.org/abs/2206.10558 • PDF: https://arxiv.org/pdf/2206.10558 |
| **Madrona / GPUDrive** | GPU-accelerated simulator **engine** (1M+ steps/s demos) | If you build your own hex-wargame, this is a path to **massive parallel** sim for HRL training | GPUDrive: https://arxiv.org/abs/2408.01584 • Engine paper: https://madrona-engine.github.io/shacklett_siggraph23.pdf |
| **RLlib (Ray)** | Distributed RL & **multi-agent** training | Out-of-the-box CTDE + population/league setups; easy to scale **self-play** | Docs: https://docs.ray.io/en/latest/rllib/multi-agent-envs.html |
| **CleanRL** | High-quality **single-file** RL baselines | Reproducible baselines; quick algorithm customization for HRL components | Paper (JMLR): https://jmlr.org/papers/v23/21-1342.html • Code: https://github.com/vwxyzjn/cleanrl |

---

## E) Board-Game TBS Analogs (imperfect-information)

| Environment | Why include it | Link |
|---|---|---|
| **Stratego (envs + DeepNash)** | Strong **turn-based**, imperfect-info benchmark; useful for opponent modeling & league evaluation | Env (community): https://github.com/JBLanier/stratego_env • DeepNash paper: https://arxiv.org/abs/2206.15378 |

---

## F) Recommended Shortlist (actionable)

1. **Start** with **Stratega** or **Tribes** for TBS scaffolding + forward models.  
2. Wrap your env with **PettingZoo (AEC)** for turn order and CTDE training; use **OpenSpiel** for exploitability/NashConv eval.  
3. Use **Griddly** (or Stratega YAML) to PCG **5×5 → 25×25** curricula.  
4. For throughput, plug **EnvPool** (and consider **Madrona/GPUDrive** if you build a custom C++/CUDA sim).  
5. For a true hex-and-counter baseline, integrate the **Wesnoth** Gym wrapper and log HRL diagnostics alongside win-rate/Elo.  

---

### Notes on HRL integration

- **Forward models** (Stratega/Tribes) let your MAXQ/OC top layer call **short lookahead** at option boundaries (HRL + MCTS hybrids).  
- **AEC turn order** in PettingZoo aligns with **HIRO** goal intervals and **Option-Critic** option commitment.  
- **PCG** (Griddly / YAML) gives principled **train/test splits** for generalization (terrain, unit mixes, fog).  
- **OpenSpiel metrics** (NashConv, exploitability) augment standard win-rate/Elo to detect **brittle policies** early.

