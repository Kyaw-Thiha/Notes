# Deterministic Finite State Automata

`DFSA` is a mathematical model of a machine that when given input string $x$, accepts or rejects $x$.
The automaton has finite set of `states`, including designated `initial state` and designated set of `accepting states`.

## Formal Definition
Formally, `DFSA` $M$ is a quintuple $M = (Q, \sum, \delta, s, F)$, where
- $Q$ is a finite `set of states`
- $\sum$ is a finite `input alphabet`
- $\delta: Q \times \sum \to Q$ is a `transition function`. ($\delta(q, a) = q'$ means there is an edge $a$ from state $q$ to state $q'$
- $s \in Q$ is the `initial state`
- $F \subset Q$ is the set of `accepting states`

## Extended Transition $f^n$
Its a `transition function` that accepts a state, go to its last occurance, and return the next state to go to in `DFSA`
$$
\delta^*: Q \times \Sigma^* \to Q
$$
In other words,
$$
\delta^{(n,u)}=\begin{cases}
n, & u=\epsilon\\[6pt]
\delta\big(\delta^{(n,v)},\,x\big), & u=vx,\ \text{ where } v\in\Sigma^{*},\ x\in\Sigma
\end{cases}
$$

## Language Accepted by M
It can be formally defined by
$$
L(M) = \{ x \in \Sigma^*: M \text{ accepted} \}
$$
where
- $M$ is the `DFSA`

Eg: $L(M) = \{ x \in \Sigma: x \text{ has odd num of 1s iff x ends with 1} \}$

### How to prove?
Use `state invariant`: ensure all checkpoints are correct

$$
\delta^*(q_{0, x}) = 
\begin{cases}
q_{0} &  
\text{iff x has even no. of 1s, x doesn't end with 1} \\

q_{1} & 
\text{iff x has odd no. of 1s, x ends with 1} \\

q_{2} & 
\text{iff x has odd no. of 1s, x ends with 0} \\

q_{3} & 
\text{iff x has even no. of 1s, x ends with 1}
\end{cases}
$$

## Conventions of Diagram
- Combine multiple state transition into single arrow separated by comma.
- Do't draw `dead state` + don't include `dead state` in `state invariant`.

