# Knapsack
Let items be $1,2,\dots,n$, where each item $t$ has
- value $v_{t}$
- weight $w_{t}$

Let $C$ be the knapsack capacity.
Decide $\forall t \in [1\dots n]$, $0 \leq x_{t} \leq 1$.
Stealing $x_{t}$ has 
- value $x_{t}v_{t}$
- weight $x_{t}w_{t}$

---
We want [[Knapsack]] $S=(x_{1},\dots, x_{n})$ $s.t.$
- $0\leq x_{t} \leq 1, \ \forall t \in [1,n)$
- $\sum_{t \in [1,n]} x_{t}w_{t} \leq C$

where 
$$
V(S) = \sum_{t} x_{t}v_{t}
$$

**Input**: $w_{t}, \ v_{t}, \ t \in [1,n]$
**Output**: Knapsack of max value.

---
## Algorithm
- Sort items in non-increasing value/weight pairs.
  (assume $\frac{v_{1}}{w_{1}} \geq \frac{v_{2}}{w_{2}} \geq \dots \geq \frac{v_{n}}{w_{n}}$)
- Steal as many whole items $1, 2, \dots, k-1$ until stealing item $k$ exceeds cap $C$.
- Steal as much o $k$ as fits knapsack.
- Steal none of $k+1, \dots, n$

[[Time Complexity|Time Complexity]]: $O(n \log n)$

---
## Proof of Correctness
Assume $\sum_{t \in [1,n]} w_{t} > C$.

`Lemma-1`: Any opt knapsack $(x_{1}, \dots, x_{n})$ satisfies $\sum x_{t} w_{t} = C$.

---
`Lemma-2`: Let $S=(x_{1}, \dots, x_{n})$ be a [[Knapsack|knapsack]] and $i < j$ $s.t.$ $x_{i} < 1$ and $x_{j} > 0$. Then, $\exists$[[Knapsack|knapsack]] $S'=(x_{1}', \dots, x_{n}')$ $s.t.$
- $V(S') \geq V(S)$ and 
- $x_{t}' = x_{t}$ $,\forall t \neq i,j$ and either $x_{i}'=1$ or $x_{j}' = 0$.

**Proof by Exchange Argument**
- $(1 - x_{i}) \ w_{i}$: Weight of remaining fraction of $i$.
- $x_{j}w_{j}$: Weight of fraction of $j$ stolen.

**Case-1**: $(1-x_{i})\ w_{i} < x_{j}\ w_{j}$
Let $S'=(x_{1}', \dots, x_{n}')$
$$
\text{where} \begin{cases}
x_{t}' = x_{t} \quad \forall t \neq i,j \\[6pt]
x_{i}' = 1 \\[6pt]
x'_{j} = x_{j} - \frac{(1-x_{i}) \ w_{i}}{w_{i}}
\end{cases}
$$

Must show that $x_{i}' \geq 0$
$$
x_{j}' = x_{j} - \frac{(1 - x_{i}) \ w_{i}}{w_{j}}
> x_{j} - \frac{x_{j}w_{j}}{w_{j}} = 0
$$

Claim: $V(S') - V(S) \geq 0$.
$$
\begin{align}
V(S') - V(S)
&= x_{i}'v_{i} + x_{j}'v_{j}  
- x_{i}v_{i} - x_{j}v_{j} \\[6pt]
&= v_{i}(x_{i}' - x_{i}) 
+ v_{j}(x_{j}' - x_{j}) \\[6pt]
&= v_{i}(1 + x_{i}) + v_{j}\left( x_{j}  
- \frac{(1-x_{i}) \ w_{i}}{w_{j}} - x_{j} \right)
\\[6pt]
&= v_{i}(1 - x_{i}) - v_{1}(1 - x_{i})  
\frac{w_{i}}{w_{j}} \\[6pt]
&= \underbrace{(1-x)}_{\geq 0 }  
\underbrace{\left( v_{i} - \frac{v_{j} w_{j}}{w_{j}} 
\right)}_{\geq 0} \text{ by (1)} \\[6pt]
&\geq 0
\end{align}
$$

$(1)$: $\frac{v_{i}}{w_{i}} \geq \frac{v_{j}}{w_{j}} \implies v_{i} \geq \frac{v_{j}w_{i}}{w_{j}}$.

---
Let $G = (x_{1}^{g}, \ x_{2}^{g}, \ x^{g}_{n})$ be a [[Knapsack|greedy knapsack]].
It is of the form $(1,\ 1, \ \dots, \ 1, \ x_{k}, \ 0, \ \dots, \ 0)$ and $\sum_{t} x_{t}^{g}w_{t} = C$.
- By `Lemma-2`, there is an [[Knapsack|optimal knapsack]] $S=(x^{s}_{1}, \ \dots, \ x^{s}_{n})$ which is of the form $(1, \ 1, \ \dots, \ 1, \ x_{l}, \ 0, \ 0, \ \dots, \ 0)$.
- By `Lemma-1`, $\sum_{t} x^{S}_{t}w_{t} = G$.

We must have $k=l$ and $x_{k} = x_{l}$.
$\therefore G=S \implies G \text{ is optimal}$.

---
