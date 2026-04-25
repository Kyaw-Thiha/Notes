# First-Order Language
A set of [[Predicate Logic|First Order formula]] is called `First-Order Language`.

As `First-Order language` consists of
- a set of variables $(x, y, z, u, v, w, \ \dots)$
- a set of predicate symbols, each with its own anity $(A, B, C)$  
- A set of constants $(a, b, c, \ \dots)$

---
`Term Definition`
A `term` is a variable or constant.
An `atomic formula` has form $A(t_{1}, \dots, t_{N})$ where 
$A$ is a `predicate symbol` with anity $n$, and each $t_{i}$ is a `term`.

---
`Definition`
The set of [[Predicate Logic|First-Order Formula]] is the smallest set $s.t.$ 
`Basis`: any atomic formula is in the set
`Induction Step`: 
If $F_{1}, \ F_{2}$ are in the set, and $x$ is a variable, then 
$\lnot F_{1}, \ (F_{1} \land F_{2}), \ (F_{1} \lor F_{2}), \ (F_{1} \to F_{2}), \ (F_{1} \leftrightarrow F_{2}), \ \exists x \ F_{1}, \ \forall x \ F_{1}$ are in the set.

---
# Parse Tree

`Parse Tree` for formula $F = \exists x \ (\forall y \ (S(x, y) \to F(y) \ ) \land \exists u \ S(u, x) \ )$

![[Parse Tree.png]]

---
`Free Variables`
A `free instance` of a var within a formula.
A variable that's not free is called `bound`.

U can check it with `parse tree` by going up from its leaf, and 
- If it encounter $\forall x$ or $\exists x$, then it is bounded within that formula
- If it doesn't (in the specific part of the tree/formula), then it is free variable

---
`Defn`
A formula without any free variables is called a `sentence`.

---
`Ques`: Is $\forall x \ A(x)$ true?
`Ans`
It depends on $3$ things:
- the `domain` $(D = R \text{ or } D = Z)$
- the definitions of the `predicates`
- the values of the `constants`

These $3$ things are called `structure`.

---
