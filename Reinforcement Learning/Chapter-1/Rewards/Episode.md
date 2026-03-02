# Episode
#rl/episode #rl/transition
An `episode` refers to a single, complete sequence of interaction between an agent and its environment.

---
## Transition
Suppose an agent starts at state $S_{1} \sim \rho \in \mathcal{M}(\mathcal{X})$.
It then chooses an action $A_{1} = \pi(S_{1})$.
Hence, it receives a reward of $R_{1} \sim \mathcal{R}(\cdot \mid S_{1}, A_{1})$.
The state is updated into $S_{2} \sim \mathcal{P}(\cdot \mid S_{1}, A_{1})$.

This single tuple $(S_{t}, A_{t}, R_{t}, S_{t+1})$ is called a `transition`.

---
## Episode
For a single episode, it is made of multiple transitions
$$
\begin{align}
(S_{1}, A_{1}, &R_{1}, S_{2}) \\[6pt]
(S_{2}, A_{2}, &R_{2}, S_{3}) \\[6pt]
&\vdots \\[6pt]
(S_{t-1}, \ A_{t-1},  & \ R_{t-1}, \ S_{t}) \\[6pt]
\end{align}
$$
 where $S_{t}$ is the `terminal state`.

Then for a new episode, the agent restarts and independantly samples another 
$$
S_{1} \sim \rho \in \mathcal{M}(\mathcal{X})
$$

---
