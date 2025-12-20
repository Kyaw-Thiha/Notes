# PDA Acceptance

`Definition`

A `PDA` $M$ accepting a string $x$ means that there's a way for $M$ to
- start with empty stack (in `initial state`)
- read all of $x$
- end in an accepting state
- end with empty block

---

`Example`

 Let $\Sigma = \{ 0, 1 \}$ and $L = \{ x \in \Sigma^*: \#_{0}(x) = \#_{1}(x) \}$
Find the `PDA` that accepts $L$.

![[PDA Example.png]]

---


