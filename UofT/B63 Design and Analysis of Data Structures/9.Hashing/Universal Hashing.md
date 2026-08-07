# Universal Hashing
A family $H$ of [[Hash Function|hash functions]] is universal iff:
For any $2$ keys $j$ and $k$ $s.t.$ $j \neq k$ and a table of size $m$,
1. At most $\frac{|H|}{m}$ functions satisfy $h(j) = h(k)$.
   $I.e:$ We randomly pick $h$ from $H$ with uniform probability.
2. $Pr(h(j) = h(k)) = \frac{1}{m}$. This is because 
$$
\begin{align}
&Pr(h(j) = h(k)) \\[6pt]
&= \frac{\# \text{ of functions that map j and k to same place}} 
{\text{Total \# of functions}} \\[6pt]
&= \frac{|H| \ / \ m}{|H|} \\[6pt]
&= \frac{1}{m} \\[6pt]
\end{align}
$$

Given a set of keys, we randomly select a [[Hash Function|hash function]] $h()$ from $H$ and use this function for every key in our set.

---
## Expected No. of Collisions
If there are $n$ keys [[Hashing|hashed]] to a [[Hash Table|table]] of size $m$, there will be $O\left( \frac{n}{m} \right)$ collisions.

**Proof**:
- Let $S$ be a set of $n$ keys.
- Let $j$ be a key not in $S$.
- Randomly pick $h$ from $H$.
- Let the $r.v$ $C$ be the number of collisions.
- For each $k \in S$, let the indicator $r.v$ $x_{k}$ be $1$ when $j$ collides with $k$ and $0$ otherwise.

$$
\begin{align}
E(c)
&= E\left( \sum_{k \in S} x_{k} \right) \\[6pt]
&= \sum_{k \in S-j} E(x_{k}) \\[6pt]
&= \sum_{k \in S-j} Pr(h(j) = h(k)) \\[6pt]
&= \sum_{k \in S-j} \frac{1}{m} \\[6pt]
&= \frac{n-1}{m} \\[6pt]
&< \frac{n}{m} \\[6pt]
&\in O\left( \frac{n}{m} \right) \\[6pt]
\end{align}
$$

---
## See Also
- [[Hashing]]
- [[Hash Function]]
- [[Hash Table]]
- [[Direct Addressing]]
- [[Closed Addressing]]
- [[Open Addressing]]
- [[Universal Hashing]]
