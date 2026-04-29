# Bellman Equation
#rl/bellman-equation/value-function 
The `Bellman Equation` for [[Value Function]] $V^{\pi}(s)$ expresses the value of a state as the sum of the immediate reward and the discounted value of the next state, calculated recursively.

![Bellman Equation|400](https://huggingface.co/blog/assets/70_deep_rl_q_part1/bellman4.jpg)

---
## Bellman Equation
For any $s \in \mathcal{S}$,
$$
V^{\pi}(s)
= r^{\pi}(s) + \gamma \int \mathcal{P}^{\pi}
(ds' \mid s, a) \ \pi(da \mid s) \ V^{\pi}(s')
$$
> The `Bellman Equation` for policy $\pi$ can be interpreted as:
> - The [[Value Function|value]] of following a policy $\pi$ starting from state $s$ 
> - is the [[Reward|reward]] that a $\pi$-following agent receives at that state 
> - plus the discounted average value that the agent receives at the next state.

### Compact Versions
Using the $P^{\pi}$ notation, we can get
$$
V^{\pi}(s)
= r^{\pi}(s) + \gamma \int \mathcal{P}^{\pi}
(ds' \mid s) \ V^{\pi}(s')
$$
or even more compactly,
$$
V^{\pi} 
= r^{\pi} + \gamma \ \mathcal{P}^{\pi} V^{\pi}
$$

---
## Deriving the Bellman Equation

**Recursive Property of Return**

Consider a sequence of rewards $R_{1}, R_{2}, \dots$ obtained by a policy $\pi$.
Recall that the [[Reward|return]] $G^{\pi}$ is a random variable defined as
$$
G^{\pi}_{t}
\triangleq \sum_{k \geq t} \gamma^{k-t} R_{k}
$$

Comparing $G_{t}^{\pi}$ and $G^{\pi}_{t+1}$, 
$$
G^{\pi}_{t}
= R_{t} + \gamma \ G^{\pi}_{t+1}
$$
This recursive structure of the return is important in [[Markov Decision Process (MDP)|MDP]].

Note that the return $G^{\pi}_{t}$ is a random variable.
So if we repeat the experiment from the same state $s$, 
- the return would be different
- but its distribution however is the same.

---
**Recursive Property of Value Function**

Hence to compute the expected value of the return,
$$
\begin{align}
V^{\pi}(s)
&= \mathbb{E}[ \ G^{\pi}_{t} \mid S_{t} = s \ ]  
\\[6pt]

&= \mathbb{E}[ \ R_{t} + \gamma G^{\pi}_{t+1}  
\mid S_{t} = s \ ] \\[6pt]

&= \mathbb{E}[ \ R(S_{t}, A_{t}) \mid S_{t}=s \ ]
+ \gamma \ \mathbb{E} [G^{\pi}_{t+1} \mid S_{t} = s] 
\\[6pt]

&= r^{\pi}(s) + \gamma \ \mathbb{E}[ \  
V^{\pi}(S_{t+1}) \mid S_{t} = s \ ]
\end{align}
$$
This also reveal the recursive nature of the [[Value Function]].

**Expanding the expected value function**

Note that $\mathbb{E}[V^{\pi}(S_{t+1}) \mid S_{t} = s]$ is the expected value of [[Value Function|value function]] $V^{\pi}(X_{t+1})$ when agent at state $s$ at time $t$ chooses `action` $A \sim \pi(\cdot \mid s)$ and get to `state` $S_{t+1}$.
We can expand this into
$$
\mathbb{E}[ \ V^{\pi}(S_{t+1}) \mid S_{t}=s \ ]
= \int \mathcal{P}(ds' \mid s, a)
\ \pi(da \mid s) \ V^{\pi}(s') 
$$
or 
$$
\mathbb{E}[ \ V^{\pi}(S_{t+1}) \mid S_{t}=s \ ]
= \sum_{s', a} \mathcal{P}(ds' \mid s, a)
\ \pi(da \mid s) \ V^{\pi}(s') 
$$

Substituting it back in, we get
$$
V^{\pi}(s)
= r^{\pi}(s) + \gamma \int \mathcal{P}^{\pi}
(ds' \mid s, a) \ \pi(da \mid s) \ V^{\pi}(s')
$$
which is the `Bellman Equation`.

---
## See Also
- [[Quality Function]]
- [[Value Function]]
- [[Bellman Equation for Quality Function]]
- [[Bellman Equation for Optimal Value Functions]]
- [[Greedy Policy]]

---