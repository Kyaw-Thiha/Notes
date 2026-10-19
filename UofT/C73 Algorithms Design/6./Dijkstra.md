# Dijkstra Algorithm

---
### Properties of Quantities being Computed
Set of nodes $R$ and $d(v),$ $\forall v \in V$ $s.t.$
$$
\forall v \in V, d(v) = \begin{cases}
\text{min weight of } s\to v \text{ path in which all nodes} \neq v  
\text{ are in } R , \ \text{ if such a path exists}
\\[6pt]
\infty , \ \text{otherwise}
\end{cases}
$$
 $$
\forall v \in R, d(v) = \delta(v)
$$

---
## Pseudocode
```python
Dijkstra(G, s, wt):
	# Initialization
	R := ∅
	d(s) := 0
	p(s) := NULL 
	for each node v != s, do:
		d(v) := ∞

	# Looping
	while R != V, do:
		u := node not in R s.t. d(u) is minimum
		R := R U {u}
		
		for each v s.t. (u, v) ∈ E, do:
			if d(v) > d(u) + wt(u, v), then:
				d(v) := d(u) + wt(u, v)
				p(v) := u
```

---
## Running Time
Note that for a graph $G=(V, \ E)$, we have $n=|V|$ and $m=|E|$.

### Naive Implementation
> **Naive Implementation**
> 
> Uses array to store $d(\cdot)$
> Time Complexity: $O(n^{2})$

### Sophisticated Implementation
> Uses the [[Extract-Min Fibonacci Heap|min-heap]].
> We then have $n$ `Extract_Min()` operations and $m$ `Change_Key()` operations.

$$
\underbrace{O(n)}_{\text{init}} + \underbrace{O \big((m + n) \log n \big)}
_{\text{while loop}}
= O \big( (m + n) \log n \big)
$$

---
## Proof
### Claims
> **Claim-1**: If $i \leq j \implies d_{i}(v) \geq d_{j}(v)$
> **Claim-2**: $u \in R_{i} - R_{i-1} \implies d_{i}(u) = d_{i-1}(u)C$
> **Claim-3**: At end of iteration $i$, if $d_{i}(v) = k \neq \infty$, $\exists$ $s\to v$ $R_{i}$-path with $wt=k$
> **Claim-4**: If $d(u) = \infty$ when $u$ is added to $R$, then $\not\exists s \to u$  path.
> **Claim-5**: $u \in R_{i} - R_{i-1} \implies d_{i}(u) = \delta(u)$. 
> At the end of the loop, $d(u) = \delta(u) \ \forall u \in R$.

---
### Proof of Claim-4
> **Contradiction**
> For the sake of contradiction, suppose the claim is false.
> Which is to say $\exists u \in R_{i} - R_{i-1}$ $s.t.$ $d_{i}(u) = \infty$ but $\exists (s\to u)$ path.
> 
> **WLOG**: Let $i$ be the earliest iteration when this happens.
> Then, $i > 1$ (since in iter-$1$, $s$ is added to $R$ and $d_{1}(s) = 0 \neq \infty$)

$$
\underbrace{s}_{s \in R}
$$

---
### Proof of Claim-5
> **Proof by complete induction on $i$**
> Suppose claim holds for all $j < i$.
> Let $u$ be the node added to $R$ in iter $i$.

> **Base Case**
> We have $i=1$, which implies that $u=s$.

> **Induction Step**
> For $i > 1$ if $d_{i}(u) = \infty$, then we are done by [[#Proof of Claim-4|Claim-4]]
> Thus, assume $d_{i}(u) \neq \infty$.
> 
> By [[#Claims|Claim-3]], we get that $d_{i}(u) \geq \delta(u)$.
> We have yet to show that $\delta_{i}(u) \leq \delta(u)$.

![image|400](https://notes-media.kthiha.com/Dijkstra/4b4343d005e20054b798f85e378e9c1e.png)

> Let $j$ be the iteration no. when $x$ was added to $R$.
> This implies that $j \leq i -1$.

$$
\begin{align}
d_{i}(u)
&= d_{i-1}(u) & & \text{[by Claim-2]} \\[6pt]
&\leq d_{i-1}(y) & & \text{[by algorithm]} \\[6pt]
&\leq d_{j}(y) & & \text{[by Claim-1]} \\[6pt]
&\leq d_{j}(x) + wt(x,y) & & \text{[by Claim-1,2 \& algorithm]} \\[6pt]
&\leq \delta(x) + wt(x,y) & & \text{[by Induction Hypothesis]} \\[6pt]
&= wt(\text{prefix of } P \text{ up to x}) + wt(x,y) 
& & \text{[by optimality substructure path]} \\[6pt]
&\leq wt(P) & & [\text{because  weight(e) >= 0} ]\\[6pt]
&= \delta(u)
\end{align}
$$

> Hence, we get that at the end of the algorithm, $d(u) = \delta(u)$ $\forall u \in R$.

---
## Dijkstra doesn't work with negative edges
Consider the following example.

![image|300](https://notes-media.kthiha.com/Dijkstra/6732e7430c92dbb4d98c4ae0dd190d05.png)

The optimal path is `A->B->C->D`, but [[Dijkstra|Dijkstra algorithm]] would only find `A->C->D`.

---