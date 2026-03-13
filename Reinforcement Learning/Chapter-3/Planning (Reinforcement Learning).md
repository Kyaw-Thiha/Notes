# Planning
#rl/planning 
In the `Planning setting`, the underlying assumption is that [[Markov Decision Process (MDP)|MDP]] is known; we know both [[Reward|reward distribution]]  $\mathcal{R}$ and [[Policy-Induced Transition Kernal|transition]] $\mathcal{P}$.

This is unlike the [[Reinforcement Learning Setting|RL Setting]] where the [[Markov Decision Process (MDP)|MDP]] is not known.
There is no direct access to $\mathcal{R}$ and $\mathcal{P}$.
Instead, the agent has only access to data coming from interaction with the environment.

---
## Methods for finding optimal policy
The methods for finding [[Markov Policy|optimal policy]] $\pi^{*}$ can be divided into $3$ categories:
- `Value-Based`
- `Direct Policy Search`
- `Hybrid Methods`

### Value-Based
These methods try to find  $V^{*}$ or $Q^{*}$ first, then use it to compute the optimal policy.
This is often done by computing the [[Greedy Policy]] $\pi_{g}$ $w.r.t$ the approximation of the [[Value Function|optimal value function]].

### Direct Policy Search
These methods directly search in the space of [[Markov Policy|policies]] without explicitly constructing the optimal value function.

### Hybrid Methods
These methods use the explicitly constructed [[Value Function|value function]] to guide an explicit search in the [[Markov Policy|policy space]].

---
## Policy Evaluation vs Control

> `Policy evaluation` refers to the problem of computing the [[Value Function|value function]] of a given [[Markov Policy|policy]] $\pi$.
>
> `Control` refers to the problem of finding the optimal value function $V^{*}$ or $Q^{*}$, or optimal policy $\pi^{*}$.

---
## Planning Algorithms

**Naive Policy Evaluation**
The naive way to evalute the optimal [[Markov Policy|policy]] $\pi$ is to brute-force.
Search through all possible states, and compute the returns at each state.
Choose the policy of actions that returns the maximum [[Reward]].

[[Naive Policy Evaluation|Read More]]

---
**Linear System of Equations**
An alternative way is to take advantage of the recursive nature of the [[Bellman Equation for Value Function]], and form a linear system of equations.
However, 
- it does not scale well due to matrix inversion
- it cannot be used for `control problem` since the equation becomes non-linear

[[Linear System Policy Evaluation|Read More]]

---
**Value Iteration**
Using the [[Uniqueness of Fixed Point]] of the [[Bellman Operator]], the value is iteratively improved to converge to optimality.
$$
\boxed{ \ 
V_{k+1} \leftarrow r^{\pi} + \gamma \
\mathcal{P}^{\pi} V_{k}
 \ }
$$
Note that this technique is applicable to both policy evaluation and control problem.

[[Value Iteration|Read More]]

---
**Policy Iteration**
Computes the optimal [[Markov Policy|policy]] $\pi^{*}$ by iteratively applying the following two steps:
- **Policy Evaluation**: Given a policy $\pi_{k}$, compute $V^{\pi_{k}}$.
- **Policy Improvement**: 
  Find a new policy $\pi_{k+1}$ that is better than $\pi_{k}$ ($V^{\pi_{k+1}} \geq V^{\pi_{k}}$).

We prove its convergence using following theorems:
1. [[Policy Improvement Theorem]]
2. [[Convergence of Policy Iteration Algorithm]]
3. [[Fast Convergence of Policy Iteration]]

---
