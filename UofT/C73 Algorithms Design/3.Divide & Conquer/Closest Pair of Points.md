# Closest Pair of Points
[[Closest Pair of Points|Closest pair of points]] aim to find the pair of points that is closer than any other points in the space.

![300](https://media.geeksforgeeks.org/wp-content/uploads/20240712111230/Closest-Pair-of-Points-between-two-points.png)

---
## Formal Definition
> **Input**: $P=\{ p_{1}, p_{2}, \dots, p_{n} \}$ of $n$ points, where $p_{i}=(x_{i}, \ y_{i})$.

We can then define distance between $p=(x,y)$ and $q=(x',y')$ as
$$
d(p,q) = \sqrt{ (x-x')^{2} + (y - y')^{2} }
$$
> **Output**: Points $p, q \in P, \ p \neq q$ such that for all $p', q' \in P$, if $p' \neq q'$ then $d(p,q) \leq d(p',q')$.

---
## Algorithm Explanation
### Main Idea
![image|500](https://notes-media.kthiha.com/Closest-Pair-of-Points/e78002ff33d3eca48d223fa6f9a18958.png)

```python
- find median m                                       O(n)
- determine L := {p : p<=m}, and R := {p: p>m}        O(n)
- recursively find closest pair in L & R
- return closest of these two & max(L), min(R)        O(1)
```

---
### Algorithm
> We can recursively find a closest pair of points $p_{L}$ and $q_{L}$ in left side of the bisector, and a closest pair $p_{R}$ and $q_{R}$ among in the right side of it.

![image|400](https://notes-media.kthiha.com/Closest-Pair-of-Points/a73c0cb2b46d65022b5c9305568204d6.png)

> **Distances**
> Let $\delta_{L} = d(p_{L}, \ q_{L})$, $\delta_{R}=d(p_{R}, \ q_{R})$ and $\delta = \min(\delta_{L}, \delta_{R})$.

> **Candidates**
> This leads us with 3 candidates to consider:
> $(a)$ $(p_{L}, q_{L})$
> $(b)$ $(p_{R}, q_{R})$
> $(c)$ $(p,q)$ within the $B = [-\delta, \delta]$ region around the bisector

> **Issue**
> It is possible that all points in $P$ are in $B$. So we would need to try all $\frac{n^{2}}{4}$ points. 
> This yields us quadratic running time again.

> **Solution**
> Sort the points in $B$ in increasing $y$-coordinate.
> Use the [[#Theorem|theorem]] that if $q$ is above $p$ and $d(p,q) < \delta$, there are at most **six** points in $B$ above $p$ and below $q$.
> So we limit our search to next **seven** points after $p$ in $B$.

---
## Full Algorithm
### Idea
> **Full Algorithm**
> 
> `Divide` with bisector in median of x-coordinate.
> `Conquer`: Recursively solve $L$ and $R$ to get candidates $(a), \ (b)$ and $\delta$.
> This gets us three candidates: $(a)$, $(b)$ and $(c)$.
> 
> Candidates $(a)$ and $(b)$ are acquired through the recursive calls on left and right sides.
> To find candidate $(c)$, we use the [[#Theorem|theorem]] trick to only check at most **seven** points above.
> 
> We then combine and compare the three candidates. 

---
### Theorem
> For any points $p, \ q \in B$ if $q$ is above $p$ and $d(p,q) < \delta$, there are **at most six** points in $B=[-\delta, \delta]$ above $p$ and below $q$.

---
### Time Complexity
By master theorem, we have
$$
T(n) = 2 \cdot T\left( \frac{n}{2} \right) + cn
$$
where
- $a=2$
- $b=2$
- $d=1$
- $a=b^{d}$

This gets us [[Time Complexity|time complexity]] of $T(n) = \Theta(n \log n)$.

---
$\forall p\neq q \in B$ $s.t.$ $\text{y-coord of}$ $p\leq$ $\text{y-coord of}$ $q$ $\text{if there are } \geq 7 \text{ points in }$ $B$ $\text{ with y-coord}$ $\text{between y-coord of }$ $p \ \& \ q \text{ then } d(p,q)\geq \delta$.

From $P$ compute:
- list $P_{x}$ of points in $P$ sorted by x-coord
- list $P_{y}$ of points in $P$ sorted by y-coord

Use $P_{x}$ to compute bisector $m$.
- $L =$ Set of points in $P$ with $\text{x-coord} < m$.
- $R =$ Set of points in $P$ with $\text{x-coord} > m$.

We are assuming that x-coordinates are distinct.

Using $P_{x}$ and $P_{y}$, we can divide $L$ into
- $L_{x}$ as sorted by $x$
- $L_{y}$ as sorted by $y$

and likewise $R$ into 
- $R_{x}$ as sorted by $x$
- $R_{y}$ as sorted by $y$

---