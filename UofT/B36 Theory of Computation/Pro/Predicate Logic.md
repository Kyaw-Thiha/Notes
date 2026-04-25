# Predicate Logic (First-Order Formula)

`Predicate`
Start with a domain (usually empty set)

`2 ways to view as`
- as a function $A.B^n \to \{ 0, 1 \}$
- as a relation $A \subset D$

`Function`
$$
A(x_{1}, x_{2}) = \begin{cases}
1 \text{ , if }(x_{1}, \dots, x_{n}) \in A \\
0 \text{ , else}
\end{cases}
$$

`Relation`
$A = \{ (x_{1}, x_{2}) \in D^N: A(x_{1}, \dots, x_{n}) = 1 \}$

`Example`
Let the domain be 
- $D = \text{set of all people}$ 
- $M(x): x \text{ is male}$
- $S(x, y): x + y \text{ be siblings}$

$$
S(x, y) \ \land \ (M(x) \leftrightarrow \lnot M(y))
$$
$$
\exists y ( \ S (xy) \land M(y) \ )
$$
