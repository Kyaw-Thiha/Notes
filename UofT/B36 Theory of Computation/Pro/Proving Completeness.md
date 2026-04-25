# Proving Completeness

`Idea`
To prove that a set $C$ of `connectives` is complete, we start with a known complete set $B$ of `connectives`.
Then, we prove that 
- $\forall F \ uoc \ B, \ \exists F' \ s.t \ F' \ uoc \ C$
- $\forall F \ uoc \ B, \ \exists F' \ s.t \ F' \ LEQV \ F$

Given any `boolean function` $f$, $B$ is complete 
Hence, $f$ can be represented by some formula $F \ uoc \ B$.

Then by what we proved, there is some formula $F'$ such that $F' \ uoc \ C$ and $F' \ LEQV \ F$.
Therefore, every `boolean function` can be represented by some formula that $uoc \ C, \text{as wanted}$.

---
`Steps`
1. Use `structural induction` to define set $G$ that $uoc \ \{ \lnot, \land \}$ or $uoc \ \{ \lnot, \lor \}$ 
2. Use `structural induction` to prove that $\forall F \in G$, $\exists F'$ $s.t.$ $F' \ uoc \ C$ and $F' \ LEQV \ F$
3. Our results follows from the fact that $\{ \lnot, \land \}$ or $\{ \lnot, \lor \}$ is complete

---
## Example

`Question`
Consider the unary connective $\underline{0}$, where $\underline{0}P$ is always falsified, regardless of whether $P$ is satisfied or falsified.

Prove that $\{ \underline{0}, \to \}$ is `complete`

`Proof`

