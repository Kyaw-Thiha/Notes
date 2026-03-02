# Observation
#rl/observation

`Observation` is the information an agent receives from the environment to describe its current situation.

---
### Formal Definitions
**Observations**
An `observation` $O_{t}$ is the variable that the agent actually observes using its various sensors.

**History**
The `history` $H_{t}$ summarizes whatever has happened to the agent up to time $t$.
$$
H_{t} = (O_{1}, A_{1}, R_{1}, \ \dots, \ O_{t-1}, A_{t-1}, R_{t-1}, \ O_{t})
$$
Given $H_{t}$, we can inquire about the probability distribution
$$
\mathbb{P}\{ O_{t+1} \mid H_{t}, A_{t} \}
$$

If we do not look at $H_{t}$, but only the current observation $O_{t}$, we can still form
$$
\mathbb{P}\{ O_{t+1} \mid O_{t}, A_{t} \}
$$
but it will have more uncertainty since we are losing information.

**Compacting the history**
Note that the size of the history gradually increases.

If we can find another variable $S_{t}$ which is compact and is a function of $H_{t}$ that satisfies
$$
\mathbb{P} \{ O_{t+1} \mid H_{t}, A_{t} \}
= \mathbb{P} \{ O_{t+1} \mid S_{t}, A_{t} \}
$$
then we can replace $H_{t}$ with $S_{t}$.

**Finding compact state for history**
Consider a dynamic system described by equation
$$
z_{t+1} = f(z_{t}, \ a_{t})
$$
where $z \in \mathbb{R}^{m}$, $a \in \mathbb{R}^{n}$, and $f: \mathbb{R}^{m} \times \mathbb{R}^{n} \to \mathbb{R}^{m}$.

Suppose $o_{t} = z_{t}$.
In this case, we don't need to keep $h_{t} = (z_{1}, a_{1}, \ \dots, \ z_{t-1}, a_{t-1}, \ z_{t})$.
The `observation` $o_{t}$ alone is enough to know whatever has happened to the system up to time $t$.

Suppose $o_{t} = g(z_{t})$ with $g: \mathbb{R}^{m} \to \mathbb{R}^{d}$.
In this case if the function $g(\cdot)$ is not a `bijective`, observation $o$ is no longer the state.
But we can still process $h_{t}$ to find a compact representation $s_{t}$.

---