# Pumping Lemma Example

> Suppose $L$ is regular.
> Let $M = (Q, \Sigma, \delta, s, F)$ be a `DFSA` s.t. $L(M) = L$ (by `Big Result`)
> Let $n = |Q|$ be states in $M$
> Let $x$ be s.t. $x \in L$ and $|x| \geq n$
> Let $q_{i} = \delta^* (s, x[: i])$ be the `current state` inside the [[Non-Deterministic Finite State Automata|DFSA]]

Then, $x = a_{1} \ a_{2} \ \dots a_{i} \ a_{i+1} \ \dots a_{j} \ a_{j+1} \ \dots\ a_{n} \ a_{n+1} \ \dots \  \ q_{|x|} \in L$