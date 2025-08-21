# Compute & Scalability Survey — Hex-Based, Turn-Based Strategy (HRL focus)

**Scope.** What it takes to train and evaluate your HRL stack (MAXQ + Option-Critic + HIRO + intrinsic motivation) as you scale from **5×5** to **25×25** hex maps with varied terrain and multi-unit control.

---

## Quick takeaways

- **Throughput first:** push environment steps/sec via vectorization (CPU) and/or GPU-native sims to keep wall-clock reasonable.  
- **Distributed off-policy beats:** leverage replay-heavy actor–learner designs (Ape-X/R2D2) or IMPALA/SEED-style decoupling for stability at scale.  
- **HRL-specific efficiency:** off-policy goal relabeling (HIRO/HER), option termination regularizers (deliberation cost), and unsupervised option pretraining (DIAYN/adInfoHRL/HIDIO) reduce data needs.  
- **Action-space engineering:** mask/factorize/branch to keep per-step compute and search branching sane in multi-unit hex TBS.  
- **Population & leagues:** PBT/self-play leagues amortize hyperparam search and harden policies under compute budgets.

---

## Core references table (compute & scalability)

| Pillar | Why it matters for your 5×5 → 25×25 hex TBS | What to borrow | Canonical paper / code |
|---|---|---|---|
| **Distributed actor–learner** | Decouple sampling from learning; keep GPUs busy while many CPU actors simulate turns. | IMPALA-style V-trace off-policy correction; multi-task training; simple scaling from single box → cluster. | IMPALA (Espeholt et al., 2018) [[paper]](https://arxiv.org/abs/1802.01561) |
| **Centralized inference** | Removes per-actor model copies & RPC overhead; better accelerator utilization when action heads are large (multi-unit). | SEED RL’s centralized model inference with many remote envs. | SEED RL (Espeholt et al., 2019/2020) [[paper]](https://openreview.net/pdf?id=rkgvXlrKwH) |
| **Single-node extreme throughput** | Useful if you don’t have a cluster: billions of frames on 1 machine. | Asynchronous PPO/APPO pipeline; GPU-batched inference; pinned memory queues. | Sample Factory (Petrenko et al., 2020) [[paper]](https://vladlen.info/papers/sample-factory.pdf) · [[repo]](https://github.com/alex-petrenko/sample-factory) |
| **High-speed env execution** | Hex TBS simulators are often the bottleneck; speeding env step dominates ROI. | Replace Python subprocess envs with **EnvPool** C++ vectorized env backend. | EnvPool (Weng et al., 2022) [[paper]](https://arxiv.org/pdf/2206.10558) |
| **GPU-native multi-agent sim** | Multi-unit per turn → large joint action space; GPU sims avoid CPU↔GPU copies and scale to thousands of concurrent battles. | Port hex TBS core to CUDA-like kernels; keep rollout + training on GPU. | WarpDrive (Lan et al., 2021/2022) [[paper]](https://arxiv.org/pdf/2108.13976) · [[JMLR]](https://www.jmlr.org/papers/volume23/22-0185/22-0185.pdf) |
| **Vectorized MARL sim** | If full GPU port is heavy, use torch-vectorized 2D physics for many concurrent scenarios (fast curriculum sweeps). | VMAS for parallel, differentiable MARL scenarios; adapt to hex micro-skirmishes. | VMAS (Bettini et al., 2022) [[paper]](https://arxiv.org/abs/2207.03530) |
| **Replay-heavy distributed Q/AC** | Sample reuse reduces environment needs (big when sim is slow); good for long horizons & partial observability. | Ape-X prioritized replay; R2D2 recurrent replay for long tactical horizons. | Ape-X (Horgan et al., 2018) [[paper]](https://arxiv.org/abs/1803.00933) · R2D2 (Kapturowski et al., 2019) [[paper]](https://openreview.net/pdf/387fb2fcee8f74c53cf707a9856f40c458f33933.pdf) |
| **Scalable replay systems** | Large hex games → big trajectories; need efficient storage/serving to learners. | Reverb for distributed replay; consider GPU-centric replay as models grow. | Reverb (Cassirer et al., 2021) [[paper]](https://arxiv.org/abs/2102.04736) · GEAR (Wang et al., 2023) [[paper]](https://arxiv.org/pdf/2310.05205) |
| **Experience replay fundamentals** | Tune replay ratio/buffers correctly to avoid instability/compute waste. | Use findings on replay ratio sensitivity; prioritized vs uniform trade-offs. | Fedus et al., 2020 [[paper]](https://proceedings.mlr.press/v119/fedus20a/fedus20a.pdf) |
| **Population-Based Training (PBT)** | Auto-tune LR/entropy/option-term costs across curriculum maps; amortizes HPO under fixed compute. | Run small population with exploit/explore, transfer weights across map sizes. | PBT (Jaderberg et al., 2017) [[paper]](https://arxiv.org/abs/1711.09846) |
| **League/self-play systems** | For adversarial enemies and meta-strategy diversity; distributes compute across opponent pools. | AlphaStar-style league training to avoid overfitting narrow strats. | AlphaStar (Vinyals et al., 2019) [[paper]](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf) |
| **IMPALA/PolyBeast/RLlib tooling** | Production-grade distributed runners reduce engineering overhead. | TorchBeast/PolyBeast for IMPALA; RLlib abstractions for actors/learners. | TorchBeast (Küttler et al., 2019) [[paper]](https://arxiv.org/abs/1910.03552) · RLlib (Liang et al., 2018) [[paper]](https://proceedings.mlr.press/v80/liang18b.html) |
| **HRL: data-efficient subgoal learning** | HRL adds extra heads/modules; off-policy goal relabeling keeps it tractable. | HIRO goal relabeling for stable off-policy hierarchical training. | HIRO (Nachum et al., 2018) [[paper]](https://arxiv.org/abs/1805.08296) |
| **Options: stable switching** | Frequent option switching wastes compute & destabilizes credit assignment. | Option-Critic for learned options; add **deliberation cost** to discourage rapid termination. | Option-Critic (Bacon et al., 2017) [[paper]](https://arxiv.org/abs/1609.05284) · Deliberation Cost (Harb et al., 2018) [[paper]](https://arxiv.org/abs/1709.04571) |
| **Unsupervised skill pretraining** | Reduces exploration burden on large maps before task rewards exist. | Pretrain DIAYN/adInfoHRL/HIDIO; plug skills as options. | DIAYN (2018) [[paper]](https://arxiv.org/abs/1806.01371) · adInfoHRL (2019) [[paper]](https://arxiv.org/abs/1906.03661) · HIDIO (2022) [[paper]](https://arxiv.org/abs/2208.09687) |
| **Action-space engineering** | Joint action grows combinatorially with units × moves; hurts search & policy nets. | **Branching DQN** factorization; **large discrete action embeddings**; **action masking**. | Action Branching (Tavakoli et al., 2018) [[paper]](https://kormushev.com/papers/Tavakoli_AAAI-2018.pdf) · Large Discrete Actions (Dulac-Arnold et al., 2015) [[paper]](https://arxiv.org/pdf/1512.07679) |
| **Tree search compute** | If you add MCTS at tactical layer, node budget dominates cost. | AlphaZero-style PUCT or more efficient hybrids to reduce self-play games. | AlphaZero (Silver et al., 2018) [[paper]](https://www.science.org/doi/10.1126/science.aar6404) · Search-Contempt MCTS (2025) [[paper]](https://arxiv.org/pdf/2504.07757) |
| **Implementation hygiene** | Correct PPO/APPO details can swing throughput and stability; saves ablation compute. | Follow modern PPO detail checklists; prefer minimal libs for clarity/perf. | PPO details (ICLR Blog Track, 2022) [[post]](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) · CleanRL (Huang et al., 2022) [[paper]](https://www.jmlr.org/papers/volume23/21-1342/21-1342.pdf) |

---

## Practical build notes for your HRL stack

- **Sim speed:** start with EnvPool for CPU vectorization; if sim becomes the bottleneck and is port-able, move core hex mechanics to GPU (WarpDrive‐style) to eliminate CPU↔GPU copies.  
- **Data plumbing:** use distributed replay (Reverb) with prioritized sampling; tune **replay ratio** and sequence lengths (R2D2-style) for long horizons in tactical layers.  
- **Hierarchical off-policy:** make **subgoal relabeling** the default (HIRO); for options, add **deliberation cost** and cap option horizons to bound compute.  
- **Action factoring:** represent per-unit moves with branching heads + **masking** from map legality to reduce wasted forward passes.  
- **Scheduling:** use a small **PBT** population across curriculum stages (tactical→operational→strategic; small→large maps) to discover hyperparam schedules without separate sweeps.  
- **Distributed runner:** if single node → **Sample Factory**; multi-node → **IMPALA/SEED** via TorchBeast/RLlib.  
- **Self-play/league (optional):** if you model enemy as learning too, adopt a lightweight **league** to maintain diverse opponents under a fixed budget.

---

## Footnotes (throughput & system facts you may cite)

- **IMPALA** introduces **V-trace** off-policy correction and scales to cluster throughput; demonstrated multi-task DMLab/Atari performance.  
- **SEED RL** centralizes inference to reduce actor lag and communication overhead at scale.  
- **Sample Factory** reports **>10⁵ FPS** training on a single multi-core node with one GPU.  
- **EnvPool** reports **~1M FPS (Atari)** / **~3M physics steps/s (MuJoCo)** on DGX-class machines, with **~2.8×** laptop gains over Python envs.  
- **WarpDrive** demonstrates **~2.9M env steps/s** with 2k envs × 1k agents on a single GPU by keeping the full loop on GPU.  

---

## Bonus: Where HRL specifically saves compute

- **Goal-conditioned off-policy (HIRO/HER):** reuses past rollouts by relabeling goals → fewer fresh environment steps to learn high-level subgoal policies.  
- **Options pretraining (DIAYN/adInfoHRL/HIDIO):** pretrain reusable maneuvers (advance/fortify/encircle) unsupervised; later fine-tune with sparse task reward.  
- **Termination regularization:** deliberation costs reduce thrashing between options, stabilizing gradients and saving unproductive rollouts.

