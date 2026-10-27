## Transitive Closure of Relation R
- $(1): R^{*}(u,u)$ $\forall u$
- $(2) : R^{*}(u,v)$ if $\exists w$ $s.t.$ $R^{*}(u,w)$ and $R(w,u)$

---
Let $G=(V,E)$ be the graph of $R$.
Let $G^{*}=(V,E^{*})$ be the graph of $R^{*}$.

Then, 
$$
(u,v) \in E^{*} \iff \exists \text{path } u\to v \text{ in } G
$$
---
$\forall i, j \in [1\dots n],$ $\forall k \in [0\dots n]$ 
$$
A(i,j,k) = \begin{cases}
\text{weight of a shortest } i\to j \text{ path} & \text{, if it exists} \\[6pt]
\infty & , \text{otherwise}
\end{cases}
$$

$$
A(i,j,k) = \begin{cases}
0 & \text{, if } k=0 \text{ and } i=j \\[6pt]
wt(i,j) & \text{, if }( k=0 \text{ and } i \neq j \text{ and } (i,j) \in E \\[6pt]
\infty & \text{, if } k=0 \text{ and } i \neq j \text{ and } (i,j) \not\in E \\[6pt]
\min\big(A(i,j,k-1), A(i,k,k-1) + A(k,j,k-1) \big) & \text{, otherwise}
\end{cases}
$$

---
## Proof
