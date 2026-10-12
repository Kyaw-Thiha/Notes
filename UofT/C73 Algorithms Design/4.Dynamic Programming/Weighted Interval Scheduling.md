# Weighted Interval Scheduling
[[Weighted Interval Scheduling|Weighted interval scheduling]] aims to find the largest subset of jobs where no two jobs conflict, and with highest total weight of chosen jobs.

![300](https://storage.googleapis.com/algodailyrandomassets/curriculum/dynamic-programming/dp6.png)

---
## Formal Definition
> **Input**: A finite set of jobs $\mathcal{J}$, along with the start time $s(j)$, finish time $f(j)$, and weight $wt(j)$ of each job $j \in \mathcal{J}$. 

> **Output**: A subset $A$ of $\mathcal{J}$ that contains no conflicting jobs and has maximum weight. That is, a set $A \subseteq \mathcal{J}$ such that
> 
> (`feasibility`) for all $j \neq j' \in A$, $j$ does not conflict with $j'$ and 
> (`optimality`) if $B \subseteq \mathcal{J}$ has the property that for all $j \neq j' \in B$, $j$ does not conflict with $j'$, then $wt(A) \geq wt(B)$.

---
Let $S = \text{opt set for jobs } 1\dots n$. Then, 
$$
S\implies
\begin{cases}
(1): \text{S is opt for jobs } [1\dots (n-1)] & n \notin S \\[6pt]
(2): S = S' \cup \{ n \} \ , S'\text{ is ot for jobs } 1\dots p(n) & n \in S
\end{cases}
$$

---
$$
\forall i \in [0 \dots n]: V = \text{value of opt set for jobs } 1\dots i 
\quad \text{(0, if i=0)}
$$
We have
$$
V(i) = \begin{cases}
\max(V(i-1), \ V(p(i)) + v(i)) &\text{, if } i>0 \\[6pt]
0 & \text{, if } i=0
\end{cases}
$$

---
## Pseudocode
### 
```python
sort jobs by finish time (f(1) <= ... <= f(n))

for i:=1 to n: do
	p(i) := max j<i s.t. f(j)<=S(i)
	(0, if no such j)

V(0) := 0
for i:=1 to n: do 
	if V(i-1) >= V(p(i)) + v(i): then
		V(i) := V(i-1)
	else: 
		V(i) := V(p(i)) + v(i)
```

---
### Time Complexity
$$
\underbrace{\theta(n \log n)}_{\text{sort}}
+ \underbrace{\theta(n \log n)}_{\text{computing } p_{i} \text{ using boundary}}
+ \underbrace{\theta(n)}_{\text{search}}
$$

---
### 
```python
S := ∅; 
i := n;
while i!=0: do
	if V(i) = V(i-1): then
		i := i - 1
	else:
		S := S ∪ {i}
		i := p(i)
return S
```

---
