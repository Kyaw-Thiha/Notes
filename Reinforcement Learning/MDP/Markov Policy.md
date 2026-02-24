# Policy
#rl/policy 
A `Markov Policy` is a decision rule used in [[Markov Decision Process (MDP)]] that maps the current state of an environment to an action.

![Markov Policy](https://gibberblot.github.io/rl-notes/_images/deterministic_vs_stochastic_policy.png)

---
**Formal Definition of Policy**
A `policy` is a sequence $\bar{\pi} = \{ \pi_{1}, \ \pi_{2}, \ \dots \}$ such that for each $t$,
$$
\pi_{t}
(a_{t} \mid X_{1}, A_{1}, \ X_{2}, A_{2}, 
\ \dots, \ X_{t-1}, A_{t-1}, \ X_{t})
$$
is a `stochastic kernal` on $\mathcal{A}$ given $\underbrace{\mathcal{X \times A \times \dots \times X \times A \times X}}_{2t-1 \text{ elements}}$.

This `kernal` satisfies
$$
\pi_{t}(\mathcal{A} \mid X_{1}, A_{1}, 
\ X_{2}, A_{2}, \ \dots, \ X_{t-1}, A_{t-1}, \ X_{t})
= 1
$$
for every $(X_{1}, A_{1}, \ X_{2}, A_{2}, \ \dots, \ X_{t-1}, A_{t-1}, \ X_{t})$.

---
**Markov Policy**
If $\pi_{t}$ is parameterized only by $X_{t}$,
$$
\pi_{t} (\cdot \mid X_{1}, A_{1}, \ X_{2}, A_{2},
\ \dots, \ X_{t-1}, A_{t-1}, \ X_{t})
= \pi_{t}(\cdot \mid X_{t})
$$
$\bar{\pi}$ is a `Markov Policy`.

**Deterministic Policy**
If for each $t$ and $(X_{t}, A_{t}, \ X_{2}, A_{2}, \ \dots, \ X_{t-1}, A_{t-1}, \ X_{t})$, 
the policy $\pi_{t}$ assigns mass one to a single point in $\mathcal{A}$, 
$\bar{\pi}$ is called a `deteministic policy`.

**Stochastic Policy**
If for each $t$ and $(X_{t}, A_{t}, \ X_{2}, A_{2}, \ \dots, \ X_{t-1}, A_{t-1}, \ X_{t})$, 
the policy $\pi_{t}$ assigns a distribution over $\mathcal{A}$,
$\bar{\pi}$ is called a `stochastic policy`.

**Stationary Policy**
If $\bar{\pi}$ is a Markov policy in the form $\bar{\pi} = (\pi, \ \pi, \ \dots)$, 
it is called a `stationary policy`.

---
