# Hierarchical Reinforcement Learning (HRL) Paradigms

Here’s an organized table of the main **HRL paradigms**, with their core idea and representative papers (linked).

| Paradigm | Short Concept | Representative Paper |
|----------|---------------|-----------------------|
| **Options Framework** | Introduces *temporally extended actions* (options), defined by initiation set, internal policy, and termination condition. | [Sutton, Precup & Singh (1999)](https://www-anw.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf) – *Between MDPs and semi-MDPs* |
| **MAXQ Decomposition** | Decomposes a task into smaller subtasks and combines their value functions in a hierarchical structure. | [Dietterich (2000)](https://www.jmlr.org/papers/volume1/dietterich00a/dietterich00a.pdf) – *Hierarchical RL with the MAXQ Value Function Decomposition* |
| **FeUdal Networks (FuN)** | Two-level agent: a manager sets abstract goals, and a worker learns low-level policies to achieve them. | [Vezhnevets et al. (2017)](https://arxiv.org/abs/1703.01161) – *FeUdal Networks for Hierarchical RL* |
| **Option-Critic Architecture** | Learns both *options* and their *policies* end-to-end, instead of hand-defining them. | [Bacon, Harb & Precup (2017)](https://arxiv.org/abs/1609.05140) – *The Option-Critic Architecture* |
| **HIRO (Off-Policy HRL)** | High-level policy proposes goals in state space; low-level executes them. Optimized for sample efficiency. | [Nachum et al. (2018)](https://arxiv.org/abs/1805.08296) – *Data-Efficient Hierarchical RL* |
| **HAC (Hierarchical Actor-Critic)** | Multi-level controllers where higher levels propose subgoals and lower levels achieve them; trained with hindsight. | [Levy et al. (2018)](https://arxiv.org/abs/1712.00948) – *Hierarchical Actor-Critic* |
| **MLS / Multilevel Subgoal Learning** | Focuses on automatic discovery of meaningful subgoals without supervision. | [Florensa et al. (2017)](https://arxiv.org/abs/1704.03012) – *Stochastic Neural Networks for HRL* |
| **HI-MAP / Goal-Conditioned HRL** | Uses *goal-conditioned policies* at multiple abstraction levels for compositional planning. | [Pong et al. (2018)](https://arxiv.org/abs/1802.09081) – *Temporal Difference Models: Model-Free Deep RL for Prediction and Control* |
| **Intrinsic Motivation HRL (e.g. adInfoHRL, HIDIO)** | Uses intrinsic rewards / mutual information to autonomously discover reusable skills (options). | [Sharma et al. (2019)](https://arxiv.org/abs/1907.01657) – *Dynamics-Aware Unsupervised Discovery of Skills*; [Xu et al. (2020)](https://openreview.net/forum?id=r-gPPHEjpmw) – *HIDIO* |
| **LLM-Guided HRL (Recent)** | Uses large language models to propose and guide subgoals/options for HRL agents. | [Li et al. (2025)](https://arxiv.org/abs/2503.19007) – *LLM-Guided Hierarchical Reinforcement Learning with Subgoal Curriculum (LDSC)* |

---
