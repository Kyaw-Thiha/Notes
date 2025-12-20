# Connectives
`Boolean Functions`
Given an integer $n>0$, a boolean function is a function that takes $n$ binary values as input and returns one binary value as output.

$$
Agreement(x, y, z) = \begin{cases}
1 & \text{if } x=y=z \\
0 & otherwise
\end{cases}
$$

`Representing Boolean`
A `propositional formula` $F$ with propositional variables $x_{1}, \dots,x_{n}$ is said to represent a boolean function $f$ of $n$ inputs $iff$, 
- $\forall \tau,$ $\tau$ satisfies $F$ whenever $f(\tau(x_{1}), \dots, \tau(x_{n})) = 1$
- $\forall \tau,$ $\tau$ falsifies $F$ whenever $f(\tau(x_{1}), \dots, \tau(x_{n})) = 0$
where $\tau$ is the `truth assignment`

---
`Completeness`
A set $C$ of connectives is said to be `complete` $\text{iff}$ every boolean function can be represented by a propositional that only uses connectives in $C$.

$\{ \lnot, \land , \lor\}$ is complete.
$\{ \lnot, \land \}$ and $\{ \lnot, \lor \}$ is complete.

`Shorthand`: "$F \ uoc \ C$" means "$F$ uses only `connectives` in $C$"

---
`Proofing Completeness`
To prove that a set $C$ of `connectives` is complete, we start with a known complete set $B$ of `connectives`.
Then, we prove that 
- $\forall F \ uoc \ B, \ \exists F' \ s.t \ F' \ uoc \ C$
- $\forall F \ uoc \ B, \ \exists F' \ s.t \ F' \ LEQV \ F$

