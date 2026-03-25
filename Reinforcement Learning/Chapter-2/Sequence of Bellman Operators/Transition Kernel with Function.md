# Transition Kernel with Function
#rl/policy/transition-kernal 

**Definition**
Given a [[Policy-Induced Transition Kernel|transition probability kernel]] $\mathcal{P}$ and a function $f \in \mathcal{B}(\mathcal{S})$, we define $\mathcal{P}f: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ as the function:
$$
\boxed{ \ (\mathcal{P}f)\ (s,a) \triangleq \int_{\mathcal{S}} 
\mathcal{P}(ds' \mid s,a) f(s') \ }
\ , \quad \forall(s,a) \in \mathcal{S} \times \mathcal{A}
$$

Likewise, given the [[Policy-Induced Transition Kernel|transition probability kernel]] induced by a [[Markov Policy|policy]] $\pi$, we define $\mathcal{P}^{\pi} f: \mathcal{S} \to \mathbb{R}$ as
$$
\boxed{ \ (\mathcal{P}^{\pi} f)(s) \triangleq 
\int_{\mathcal{S}} \mathcal{P}^{\pi}(ds' \mid s)
f(s') \ } \ , \quad \forall s \in \mathcal{S}
$$

$\mathcal{P}^{\pi}f$ can be thought of as a function whose value at state $s$ is the expected value of function $f$ according to the distribution $\mathcal{P}^{\pi}(\cdot \mid s)$.
$$
(\mathcal{P}^{\pi}f)(s) = \mathbb{E}_{S' \ \sim \ 
\mathcal{P}^{\pi}(\cdot \mid s)} 
[ \ f(S') \ ]
$$

---
## See Also
- [[Policy-Induced Transition Kernel]]
- [[Markov Policy]]