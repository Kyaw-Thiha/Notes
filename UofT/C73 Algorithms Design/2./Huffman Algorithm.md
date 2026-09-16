# Huffman Algorithm
Let $\Gamma$ be the alphabet with $|\Gamma| \geq 2$.
Code for $\Gamma$ be a function from $\Gamma$ to $\{ 0,1 \}^{*}$.

E.g: ASCII fixed-length code of $[ \ \log_{2}(|\Gamma|) \ ]$ bits per symbol.
In here, same $8$ bits for letters that may not be used that often.
So instead, [[Huffman Algorithm|Huffman algorithm]] gets us variable-length code.

---
Suppose the codes to be
- $A$: $1$
- $B$: $01$
- $C$: $010$

For **unique decoding**: prefix-free codes.
No codeword is a prefix of another.

---
## Optimal Code
**Input**: $\Gamma$, $\forall x \in \Gamma$, $f(x) = \text{freq of } x$ $s.t.$ $\sum f(x) = 1$.
**Output**: [[AVL Tree|Binary tree]] representing optimal prefix-free code.
$\text{I.e:}$ One that minimizes.
$$
\text{AD}(T)
= \sum_{x \in \Gamma} f(x) 
\underbrace{\text{depth}_{T}(x)}_{
\text{depth of leaf } x \text{ in T}}
$$

---
## Observations
- An optimal tree must be a full tree. 
- If $x, \ y$ are symbols with minimum frequency, then $\exists$ an optimal tree $s.t.$ $x \ \& \ y$ are siblings with max depth.

---
- Create nodes with leaves for all the symbols.
- Find $x, y = \text{min freq symbls}$.
- Remove $x$ & $y$ from $\Gamma$ and replace them by $z$(new symbol) with frequency $f(x) + f(y)$.
- Recursively find optimal code for smaller set of symbols with the new frequency.

---
## Program Correctness
> **Theorem**: Algorithm produces optimal tree.

Proof by induction on $n = |\Gamma|$. If $|\Gamma| = n$, then for any frequency of symbols in $\Gamma$, the algorithm produces an optimal tree.

**Basis**: $n=2$
**Induction Step**: Assume trace by $n$ symbols.
- Let $|\Gamma| = n+1$ with frequency $f$.
- $H = \text{tree produced by algo}$ for $(\Gamma, f)$
- $x,y = \text{two least freq symbols merged first by algo}$

$$
\Gamma' = (\Gamma - \{ x,y \}) \cup \underbrace{\{ z \}}_{\text{new}}
$$
and
$$
f'(a)
= \begin{cases}
f(a) & \forall a \in \Gamma - \{ x,y \} \\[6pt]
f(x) + f(y) & a = z
\end{cases}
$$

---
![image|150](https://notes-media.kthiha.com/Huffman-Algorithm/42dfa354ed431ab398adafdd772db87f.png)
This is original tree $H$.
$$
\text{AD}(H) = \text{AD}(H') + f(x) + f(y)
$$

![image|150](https://notes-media.kthiha.com/Huffman-Algorithm/74dea4a83935e415268b1aaf5072685e.png)
This is $H'$ as produced by algo on $(\Gamma', f')$ .
$$
|\Gamma'| = n
\implies H' \text{ is optimal(by IH)}
$$

---
Let $T$ be the optimal tree
![image|150](https://notes-media.kthiha.com/Huffman-Algorithm/76befea788343f4c9f0830ca298762e9.png)

![image|150](https://notes-media.kthiha.com/Huffman-Algorithm/5778ed9710dba979c67cf4198e3c6414.png)
$$
\begin{align}
\text{AD(H)}
&= \text{AD}(H') + f(x) + f(y) \\[6pt]
&\leq \text{AD}(\Gamma') + f(x) + f(y) \\[6pt]
&= \text{AD}(T)
\end{align}
$$

---
