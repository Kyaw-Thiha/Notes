# Complexity Proof
Here are good techniques to remember to prove [[time complexity]].

---
### Proving Big-$O$
Recall that in [[Big-O]], we are trying to upper bound.
- First, remove negative terms.
- For positive terms, convert to the to the power of $n^{t}$ where $t$ is the $g(n) = O(n^{t})$. We can do this since we don't need tight bound.

[[Big-O|Read more]]

---
### Proving Big-$\Omega$
Recall that in [[Big-Omega]], we are trying to lower bound.
- First, remove positive terms.
- For negative terms, factorize out $n^{t}$. 
  Then, choose $n_{0}$ such that the term is positive.
  Set the coefficient of $n^{t}$ as $c$.

[[Big-Omega|Read more]]

---
### Using limits
Recall that to prove with [[Using limits for Big-O|limits]], 
- you try to find finite limit of $\lim_{ n \to \infty } \frac{f(n)}{g(n)}$.

Recall that to disprove with [[Using limits for Big-O|limits]],
- you show that limit of $\lim_{ n \to \infty } \frac{f(n)}{g(n)}$ is infinite.

**Tips**:
- Use the l'hopital rule to simplify limits.
- $f(n) = \Omega(g(n)) \iff g(n) = O(f(n))$

---
#### Caveats
Proving with [[Using limits for Big-O|limits]] does not work when limit does not exist, but is not infintie.

Example would be oscillitating functions like piecewise and sinusoidal functions.

In such situation, 
- use sin/cos amplitude to bound.
- or if piecewise, manually bound each cases.

[[Using limits for Big-O|Read more]]

---
## See Also
- [[Big-Theta]]
- [[Big-O Example]]
- [[Big-O]]
- [[Big-Omega]]
- [[Using limits for Big-O]]
- [[Exponential Rules]]
- [[Logarithmic Rules]]
