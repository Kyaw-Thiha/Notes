# Temporal Difference Error
The term
$$
\boxed{ \ \delta_{t} \triangleq R_{t} + \gamma 
V(S_{t+1}) - V(S_{t}) \ }
$$
is called the `temporal difference error`.
> This is a noisy measurement of how close we are to $V^{\pi}$.

---
To see this clearly, let's define the dependence of the `TD error` on its components more explicitly.
Given a [[Transition Kernel with Function|transition]] $(S,A,R,S')$ and a value function $V$, define
$$
\delta(S,R,S'; \ V)
\triangleq R + \gamma V(S') - V(S)
$$
We then have
$$
\mathbb{E}[\delta(S,R,S'; \ V) \mid S=s]
= (T^{\pi}V)(s) - V(s)
= BR(V)(s)
$$
So in expectation, the `TD Error` is equal to the [[Bellman Error|Bellman residual]] of $V$, evaluated at state $s$.

Recall that when Bellman residual is zero when $V=V^{\pi}$.
So when we are at or close to $V^{\pi}$, the TD error is close to zero in expectation.

---
## See Also
- [[Temporal Difference Learning for Policy Evaluation(TD)]]
- [[Bellman Error]]
- [[TD Learning for Action-Value Function]]