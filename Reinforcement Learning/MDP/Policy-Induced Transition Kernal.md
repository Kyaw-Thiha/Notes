# Policy-Induced Transition Kernal
#rl/policy/transition-kernal 
A `policy-induced transition kernal` represents the probability distribution of the next state $s'$ given the current state $s$.

---
**Formal Definition**
An agent is following a [[Markov Policy|Markov stationary policy]] $\pi$ whenever $A_{t}$ is selected according to the policy $\pi(\cdot \mid X_{t})$.
$$
\begin{align}
&\text{Deterministic}: A_{t} = \pi(X_{t}) \\[6pt]
&\text{Stochastic}: A_{t} \sim \pi(\cdot \mid X_{t})
\end{align}
$$
The policy $\pi$ then induces two `transition probability kernals`:
- $\mathcal{P}^{\pi}: \mathcal{X} \to \mathcal{M}(\mathcal{X})$
- $\mathcal{P}^{\pi}: \mathcal{X \times A} \to \mathcal{M}(\mathcal{X \times A})$

For subset $S \subset \mathcal{X}$ and deterministic policy $\pi$, we can denote
$$
(P^{\pi})(S \mid s)
\triangleq \int_{\mathcal{X}}
\mathcal{P} (ds' \mid s, \ \pi(s))
\ \mathbb{I}_{\{ s' \in S \}}
$$

For subset $C \subset \mathcal{X \times A}$ and deterministic policy $\pi$, we can denote
$$
(P^{\pi})(C \mid s, a)
\triangleq \int_{\mathcal{X}}
\mathcal{P} (ds' \mid s, a)
\ \mathbb{I}_{\{ \ (s', \pi(s')) \ \in \  C \ \}}
$$

---
**Countable State-Action Space**
When we have countable state-action space, we can use summation instead of integrals.

For example, 
$$
\begin{align}
&(\mathcal{P}^{\pi}) (\mathcal{S} \mid x) \\[6pt]
&\triangleq \sum_{s' \in \mathcal{X}}  
\mathcal{P}(s' \mid s, \pi(s))  
\ \mathbb{I}_{\{ s' \in S \}} \\[6pt]
&= \sum_{s' \in \mathcal{S}}
\mathcal{P}(s' \mid s, \pi(s))
\end{align}
$$
So for particular $s' \in \mathcal{X}$, we have
$$
(\mathcal{P}^{\pi})(s' \mid s)
= \mathcal{P}(s' \mid s, \pi(s))
$$

---
**Stochastic Policy**
If policy $\pi$ is stochastic, we have
$$
(P^{\pi})(S \mid s)
\triangleq \int_{\mathcal{X}}
\mathcal{P} (ds' \mid s, \ \pi(s))
\ \pi(da \mid s)
\ \mathbb{I}_{\{ s' \in S \}}
$$
and
$$
(P^{\pi})(C \mid s, a)
\triangleq \int_{\mathcal{X}}
\mathcal{P} (ds' \mid s, a)
\ \pi(da' \mid s')
\ \mathbb{I}_{\{ \ (s', \pi(s')) \ \in \  C \ \}}
$$

---
**Extending to m-steps**
We can extend the definition of $\mathcal{P}^{\pi}$ to follow a policy for $m\geq 1$ steps inductively.

For $S \subset \mathcal{X}$,
$$
(P^{\pi})(S \mid s)
\triangleq \int_{\mathcal{X}}
\mathcal{P} (ds' \mid s, \ \pi(s))
\ (\mathcal{P}^{\pi})^{m-1}
\ (S \mid s)
$$
and for $C \subset \mathcal{X \times A}$, 
$$
(P^{\pi})(C \mid s, a)
\triangleq \int_{\mathcal{X}}
\mathcal{P} (ds' \mid s, a)
\ (\mathcal{P}^{\pi})^{m-1}
\ (B \mid s', \pi(s'))
$$

---
