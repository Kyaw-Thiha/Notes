# Fractional Knapsack
In [[Fractional Knapsack|fractional knapsack problem]], we aims to maximize the value of objects we can carry in a bag.

![|400](https://mwofqqzjzhhinqamxmxt.supabase.co/storage/v1/object/public/blog-images/88d13d47-10d2-45dc-a734-6b89b6200de0/fractional-knapsack-problem-greedy-approach-guide-1775814454582.png)

---
## Formal Definition
A [[Fractional Knapsack|knapsack]] is a vector $S=(x_{1}, \dots,x_{n})$ that satisfies the following `feasibility constraints`:
- $(a)$ for each item $t$, $0 \leq x_{t} \leq 1$
- $(b)$ $\sum ^{n}_{t=1} w_{t} x_{t} \leq C$. The total weight of the fractions of stolen items does not exceed the capacity of the knapsack.

$S$ is an `optimal knapsack` if it is a [[Fractional Knapsack|knapsack]] of maximum value. For all knapsacks $S'$, $V(S) \geq V(S')$.

**Input**: $C>0$ and $v_{t} > 0$, $w_{t} > 0$ for each $t \in [1 \dots n]$.
**Output**: An optimal knapsack $\max V(S) = \sum_{t} x_{t}v_{t}$.

---
## Algorithm
- Sort items in non-increasing value/weight pairs.
  (assume $\frac{v_{1}}{w_{1}} \geq \frac{v_{2}}{w_{2}} \geq \dots \geq \frac{v_{n}}{w_{n}}$)
- Steal as many whole items $1, 2, \dots, k-1$ until stealing item $k$ exceeds cap $C$.
- Steal as much of $k$ as fits knapsack.
- Steal none of $k+1, \dots, n$

[[Time Complexity|Time Complexity]]: $O(n \log n)$

---
## Proof of Correctness
Assume $\sum_{t \in [1,n]} w_{t} > C$.


---
### Lemma-1
Let $S=(x_{1}, \dots, x_{n})$ be a [[Fractional Knapsack|knapsack]] and $i < j$ $s.t.$ $x_{i} < 1$ and $x_{j} > 0$. 
Then, there exists [[Fractional Knapsack|knapsack]] $S'=(x_{1}', \dots, x_{n}')$ $s.t.$
- $\mathcal{V}(S') \geq \mathcal{V}(S)$ and 
- $x_{t}' = x_{t}$ $,\forall t \neq i,j$. $S'$ is identical to $S$ in all items except $i$ and $j$. 
  And either $x_{i}'=1$ or $x_{j}' = 0$.

#### Proof by Exchange Argument
We can first define weight of fractions as
- $(1 - x_{i}) \ w_{i}$: Weight of remaining fraction of $i$.
- $x_{j}w_{j}$: Weight of fraction of $j$ stolen.

---
#### Case-1
> We can transfer to item $i$ all of the weight occupied by item $j$.
> 
> $\boxed{ \ (1-x_{i})\ w_{i} < x_{j}\ w_{j} \ }$

Then, let $S'=(x_{1}', \dots, x_{n}')$ where
$$
\text{where} \begin{cases}
x_{t}' = x_{t} \quad \forall t \neq i,j \\[6pt]
x_{i}' = 1 \\[6pt]
x'_{j} = x_{j} - \frac{(1-x_{i}) \ w_{i}}{w_{i}}
\end{cases}
$$

First, we must prove that $S'$ is indeed a [[Fractional Knapsack|knapsack]] and that it satisfies constraints $(a)$ and $(b)$.

> **Proving feasibility $(a)$**:
For $(a)$, we must show that $0 \leq x_{i}' \leq 1$. So, first we show that $x'_{i} \geq 0$:
$$
x_{j}' = x_{j} - \frac{(1 - x_{i}) \ w_{i}}{w_{j}}
> x_{j} - \frac{x_{j}w_{j}}{w_{j}} = 0
$$
> and then show that $x'_{i} \leq 1$:
$$
x'_{i} = x_{i} + \left( \frac{w_{j}}{w_{i}} \right) x_{j}
\leq x_{i} + \frac{w_{i} \ (1 - x_{i})}{w_{i}} = 1
$$

> **Proving feasibility $(b)$**
> By definition of $x_{i}'$  and $x_{j}'$, we have that

$$
\sum_{1 \leq t \leq n} w_{t} x'_{t}
= \left( \sum_{t \neq i,j} w_{t} x'_{t} \right) + w_{i}x'_{i} + w_{j} x'_{j}
= \left( \sum_{t \neq i,j} w_{t} x'_{t} \right) + w_{i} \left( x_{i} + \frac{w_{j}}{w_{i}} x_{j} \right)
= \sum_{1 \leq t \leq n} w_{t} x_{t} \leq C
$$

> **Claim**: $\mathcal{V}(S') \geq \mathcal{V}(S)$. 
> Using  definition of $x'_{i}$ and $x_{j}'$, we can prove that $V(S') - V(S) \geq 0$.

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

where $(1)$: $\frac{v_{i}}{w_{i}} \geq \frac{v_{j}}{w_{j}} \implies v_{i} \geq \frac{v_{j}w_{i}}{w_{j}}$.

---
#### Case-2
> We can only transfer part of the weight occupied by item $j$ before exceeding the remaining weight of item $i$.
> 
> $\boxed{ \ w_{j}x_{j} > w_{i}(1-x_{i}) \ }$

Let $S' = (x'_{1}, \dots, x'_{2})$ where
$$
\begin{align}
x'_{t} &= x_{t}, \quad \forall t\neq i,j \\[6pt]
x'_{t} &= 1 \\[6pt]
x'_{j} &= x_{j} - \left( \frac{w_{i}}{w_{j}} \right) (1-x_{i})
\end{align}
$$
Analogous to [[#Case-1|Case-1]], we can verify that $S'$ is a [[Fractional Knapsack|knapsack]] and that $\mathcal{V}(S') \geq \mathcal{V}(S)$.

---
### Lemma-2 
> Any optimal knapsack $S = (x_{1}, \dots, x_{n})$ satisfies $\sum ^{n}_{t=1} x_{t} w_{t} = C$.

#### Proof
> **Proof by Contrapositive**
> Suppose $\sum ^{n}_{t=1} w_{t}x_{t} \neq C$. So $S$ has some surplus capacity $c = C - \sum ^{n}_{t=1} w_{t}x_{t} > 0$.
> Since $\sum ^{n}_{t=1} w_{t} > C$, it cannot be that $x_{t}=1$ for all items $t$. 
> So there is some item $k$ such that $x_{k} < 1$.

> By increasing $x_{k}$ to $x_{k}+\min\left( \frac{c}{w_{k}}, 1 - x_{k} \right)$, we obtain a [[Fractional Knapsack|knapsack]] of greater value than $S$.
> So, $S$ is not optimal.

---
### Lemma-3 (Correctness)
> The [[Fractional Knapsack|knapsack]] $G=(x_{1}^{g}, \dots, x_{n}^{g})$ returned by the [[Greedy Algorithm|greedy algorithm]] satisfies the following:
> (a): There is some item $k$ such that $x^{g}_{t}=1$ for all items $t < k$, and $x_{t}^{g}=0$  for all items $t > k$.
> (b): $\sum^{n}_{t=1} w_{t}x_{t}^{g} = C$

Let $G = (x_{1}^{g}, \ x_{2}^{g}, \ \dots, \ x^{g}_{n})$ be a [[Fractional Knapsack|greedy knapsack]].
It is of the form $(1,\ 1, \ \dots, \ 1, \ x_{k}, \ 0, \ \dots, \ 0)$ and $\sum_{t} x_{t}^{g}w_{t} = C$.
- By [[#Lemma-1]], there is an [[Fractional Knapsack|optimal knapsack]] $S=(x^{s}_{1}, \ \dots, \ x^{s}_{n})$ which is of the form $(1, \ 1, \ \dots, \ 1, \ x_{l}, \ 0, \ 0, \ \dots, \ 0)$.
- By [[#Lemma-2]], $\sum_{t} x^{S}_{t}w_{t} = G$.

We must have $k=l$ and $x_{k} = x_{l}$.
$\therefore G=S \implies G \text{ is optimal}$.

---
## See Also
- [[Greedy Algorithm]]
- [[Interval Scheduling]]
- [[Minimum Lateness Scheduling]]
- [[Fractional Knapsack]]
- [[Huffman Algorithm]]
