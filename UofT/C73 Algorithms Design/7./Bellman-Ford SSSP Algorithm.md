# Bellman-Ford SSSP Algorithm
For all $u \in V,$ for all $k \in \mathbb{N}$ (actually just $k \in [0\dots n]$)
$$
S(u,k) = \begin{cases}
\text{weight of shortest } s\to u \text{ k-path} & \text{, if such path exists}  
\\[6pt]
\infty & , \text{otherwise}
\end{cases}
$$

---
- $(1):$ $S(u,k) \leq S(u,k-1)$
- $(2):$ $S(u,k) \neq S(u,k-1) \iff s\to u \text{ k-paths have k edges}$ 

$$
S(u,k) = \begin{cases}
0 & \text{if } k=0 \text{ \& } u=s \\[6pt]
\infty & \text{if } k=0 \text{ \& } u\neq s \\[6pt]
\min\big(\{ S(u, k-1) \} \cup \{ S(v,k-1) + wt(v,u): (v,u) \in E \} \big)
\end{cases}
$$
where S-label of $u$ in phase $k$.

---
## Lemma
> **Lemma: 4.7**
> 
> If $\forall u \in V, \ S(u,k) = S(u, k-1) \implies \forall u \in V, \ S(u,k+1) = S(u,k)$.

Proof is as follows
$$
\begin{align}
S(u,k)  
&= \min\big( \{ S(u,k-1) \} \cup \{ S(v,k-1) + wt(v,u): (v,u) \in E \} \big) \\[6pt]
&= \min\big( \{ S(u) \} \cup \{ S(v) + wt(v,u): (v,u) \in E \} \big) \\[6pt]
&= S(u,\ k+1) \\[6pt]
\end{align}
$$

---
## Corrolary
> **Corrolary 4.8**
>
> If `S-labels` do not change in phase $k$, they never change in phase $k' \geq k$, and $S(u,k) = \text{wt of a shortest } s\to u \text{ path}$.

---
## Lemma
> **Lemma 4.9**
>
> If for some $u \in V$, $S(u,n) \neq S(u,n-1)$, then every shortest $s\to u$ $n$-path $p$:
> $(a)$ contains a cycle, and
> $(b)$ every cycle on $p$ has a negative weight.

---
## Pseudocode
```python
BellmanFord(G, s, wt):
	S(s,0) := 0
	for each u ∈ V - {s} do: 
		S(u,0) := ∞
	
	for k := 1 to n do:
		S_changed = false
		for each u ∈ V, do:
			S(u,k) := S(u, k-1)
			for each v ∈ V such that (v,u) ∈ E do
				if S(u,k) > S(v, k-1) + wt(v,u), then:
					S(u,k) := S(v, k-1) + wt(v,u)
					S_changed := true
		
		if S_changed = false, then
			break
	if S_changed = false, then:
		for each u ∈ V do 
			δ(u) := S(u,k)
		return δ
	else:
		reuturn "negative-weight cycle"
```