# Johnson Algorithm

Adjust edge weights from $wt$ to $\hat{wt}$.
$(a)$ $\hat{wt}(e) \geq 0,$ and
$(b)$ preserve shortest paths

---
Define node weights $x_{u},$ $\forall u \in V$.
$$
\forall (u,v) \in E : \hat{wt}(u,v) = wt(u,v) + x_{u} - x_{v}
$$
Let $p=u_{1}, \ u_{2}, \ \dots, \ u_{k}$ be a path in $G$.
$$
\begin{align}
\hat{wt}(p) = & \ \hat{wt}(u_{1}, u_{2}) + \dots + \hat{wt}(u_{k-1}, u_{k}) \\[6pt]
= & \ wt(u_{1}, u_{2}) + x_{u_{1}} - x_{u_{2}} \\[6pt]
& \ + wt(u_{2}, u_{3}) + x_{u_{2}} - x_{u_{3}} \\[6pt]
& \ + wt(u_{3}, u_{4}) + x_{u_{3}} - x_{u_{4}} \\[6pt]
& \quad \quad \quad \quad \quad \vdots  \\[6pt]
& \ + wt(u_{k-2}, u_{k-1}) + x_{u_{k-2}} - x_{u_{k-1}} \\[6pt]
& \ + wt(u_{k-1}, u_{k}) + x_{u_{k-1}} - x_{u_{k}} \\[6pt]
= & \ wt(p) + x_{u_{1}} - x_{u_{k}}
\end{align}
$$

Hence, we preserve the shortest paths.

---
For $(a)$, we want $\forall(u,v) \in E$, 
$$
\begin{align}
\hat{wt}(u,v) = & \ wt(u,v) + x_{u} + x_{v} \geq 0 \\[6pt]
\iff & \ x_{v} - x_{u} \leq wt(u,v)
\end{align}
$$
The difference constraints is associated with $G$ and $wt$.

---
![image|300](https://notes-media.kthiha.com/Johnson/5f5c8eba60f2079db6cd6330b026ec4b.png)

We have
- $x_{B} - x_{A} \leq -4$
- $x_{C} - x_{A} \leq -2$
- $x_{C} - x_{B} \leq 1$ 
- $x_{D} - x_{C} \leq 1$
- $x_{A} - x_{D} \leq 3$

One way that we could have is
- $x_{A} = 0$
- $x_{B} = -4$
- $x_{D} = -2$
- $x_{C} = -3$

---
From $G = (V,E), \ wt$ construct $G_{S} = (V_{S}, \ E_{S})$ and $wt_{S}$.
$$
\begin{align}
&V_{S} = V \cup \{ S \} \\[6pt]
&wt_{S}(u,v) = \begin{cases}
0 , & \text{, if } u=s \\[6pt]
wt(u,v) & \text{, otherwise}
\end{cases}
\end{align}
$$

---
> **Theorem 4.10**
> $(a)$ If the difference constraints associated with $G$ $\&$ $wt$ are satisfiable, then $G$ has no negative weight cycles.
> $(b)$ If $G$ has no negative weight cycle, then taking $x_{u} = \text{wt of shortest } s\to u \text{ path in } G_{s} , \ \forall u \neq s$ satisfies the difference constraints.

---
### Proof-(a)
Let $C = u_{1}, u_{2}, \dots, u_{k}$ be any cycle in $G$, and let $x^{*}_{u}$ $\forall u \in V$ satisfy the difference constraints.
Then, we have
$$
\begin{align}
\chi^{*}_{u_{2}} - \chi^{*}_{u_{1}} &\leq wt(u_{1}, u_{2}) \\[6pt]
\chi^{*}_{u_{2}} - \chi^{*}_{u_{3}} &\leq wt(u_{2}, u_{3}) \\[6pt]
\chi^{*}_{u_{3}} - \chi^{*}_{u_{4}} &\leq wt(u_{3}, u_{4}) \\[6pt]
&\ \ \ \vdots \\[6pt]
\chi^{*}_{u_{k}} - \chi^{*}_{u_{k-1}} &\leq wt(u_{k}, u_{k-1}) \\[6pt]
\end{align}
$$
All of these cancel out to get $wt(C) \geq 0$.

---
### Proof-(b)
For $(b)$, $\forall w \in V,$ let $p_{w}$ be the shortest $s\to w$ path in $G_{S}$.
Let $\chi^{*}_{w} = wt(p_{w})$. We claim that
$$
\begin{align}
\forall (u,v) \quad & \ \chi^{*}_{v} - \chi^{*}_{u} \leq wt(u,v) \\[6pt]
\iff & \ \chi^{*}_{v} \leq \chi^{*}_{u} + wt(u,v)
\end{align}
$$

---
## Pseudocode
```python
Johnson(G, wt):
	construct G_s and wt_s from G and wt
	r := BellmanFord(G_s, s, wt_s)
	if r = "negative-weight cycle", then
		return r
	else: 
		for each node u of G, do
			x^{*}_u := r(u)
		for each edge (u,v) of G, do
			\hat{wt}(u,v) := wt(u,v) + x^{*}_u - x^{*}_v
		for each node u of G, do
			δ := Dijkstra(G, u, \hat{wt})
			for each node v of G, do
				Δ(u,v) := δ(u,v) + x^{*}_v - x^{*}_u
		return Δ
```

---
> We can use BellmanFord to solve the difference constraints.

---