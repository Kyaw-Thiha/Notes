# Pushdown Automata 
Informally, a `PDA` is an [[Non-Deterministic Finite State Automata|NFSA]] with a stack.

`Definition`

Formally a `PDA` is a $6$-tuple $M = (Q, \sigma, \tau, \sigma, s, F)$ where
- $Q$: set of `states`	
- $\Sigma$: Input alphabet
- $\tau$: Stack alphabet
- $\delta$: `transition function`
- $s$: initial state ($s \in Q$)
- $F$: set of accepting states $(F \subset Q)$
---

`Transitions`
$$
\delta: Q 
\times (\Sigma \ U \{ \epsilon \}) 
\times(\tau \ U \{ \epsilon \}) 
\to
P(Q \times (\tau \ U \{ \epsilon \}))
$$
- $X = \epsilon$, $Y \in \tau$ $\to$ push $Y$ on stack
- $X \in \tau$, $Y = \epsilon$ $\to$ pop $Y$ from stack (only when $X$ is at top of stack)
- $X,Y \in \tau$ $\to$  replace $X$ by $Y$ at top of stack (only when $X$ is at top of stack)
- $X = \epsilon = Y$ $\to$ stack operation

---