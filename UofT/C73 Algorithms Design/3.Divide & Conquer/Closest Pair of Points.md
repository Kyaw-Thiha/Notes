# Closest Pair of Points
**Input**: $P=\text{set of n points (on plane)}$, with each point $p=(x,y)$.
We can then define distance between $p=(x,y)$ and $q=(x',y')$ as
$$
d(p,q) = \sqrt{ (x-x')^{2} + (y - y')^{2} }
$$
**Output**: $p\neq q, \ p,q \in P$ $s.t.$ $d(p,q) \leq d(p',q')$ $\forall p'\neq q' \in P$.

---
## Algorithm
![image|500](https://notes-media.kthiha.com/Closest-Pair-of-Points/e78002ff33d3eca48d223fa6f9a18958.png)

```python
- find median m                                       O(n)
- determine L := {p : p<=m}, and R := {p: p>m}        O(n)
- recursively find closest pair in L & R
- return closest of these two & max(L), min(R)        O(1)
```

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