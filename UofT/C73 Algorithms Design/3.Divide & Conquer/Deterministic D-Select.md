# Deterministic D-Select
Assume $a_{1}, \dots, a_{n}$ are distinct.
$$
(a_{i}, i) < (a_{j}, j)
$$

![image|300](https://notes-media.kthiha.com/Selection-Algorithm/08dc88b3b23bda2933953ffb82a6d112.png)

**Input**: $\text{Seq A} = a_{1}, \dots, a_{n}; \ k\in[1\dots n]$
**Output**: $k^{th} \text{ smallest element in } A$

---
## Algorithm
```python
CHOOSE_SPLITTER(A):
	Partition A into [n/5] groups of 5 elements
	(with <= 4 elements left over)
	Find median of each group (by sorting)
	M := list of medians
	return SELECT(M, [M/2])
```

---
## Running Time
$$
T(n) = \underbrace{T\left( \left[ \frac{n}{5} \right] \right)}
_{1st \text{ call to Select}}
+ \underbrace{T\left( \left[ \frac{3n}{4} \right] \right)}
_{2nd \text{ call to SELECT}}
+ cn
$$
**Claim**: $T(n) \leq dn$, where $d$ is some constant.

$$
\begin{align}
T(n) &= T\left( \left[ \frac{n}{5} \right] \right)  
+ T\left( \left[ \frac{3n}{4} \right] \right) + cn \\[6pt]
&\leq d\left[ \frac{n}{5} \right] + d\left[ \frac{3n}{4} \right] + cn \\[6pt]
&\leq \frac{dn}{5} + \frac{3dn}{4} + cn \\[6pt]
&\leq \frac{19dn}{20} + cn \\[6pt]
\end{align}
$$
We can then get
$$
\begin{align}
&\frac{19dn}{20} + cn \leq dn \\[6pt]
\implies &\frac{dn}{20} \geq cn \\[6pt]
\iff & d \geq 20c
\end{align}
$$

---
Let 
- $|A^{-}| \geq 3 \cdot |\frac{\left[ n/5 \right]}{2}|$ 
- $|A^{+}| < 3 \cdot |\frac{\left[ n/5 \right]}{2}|$ 
- no. of elements in recursive call $\leq n-3\left[ \frac{\left[ \frac{n}{5} \right]}{2} \right]$.

$$
\begin{align}
n - 3 \cdot \left[ \frac{\left[ \frac{n}{5} \right]}{2} \right] 
&\leq n - 3 \cdot \frac{\left[ \frac{n}{5} \right]}{2} \\[6pt]
&\leq n - 3 \cdot \frac{\left[ \frac{n-4}{5} \right]}{2} \\[6pt]
&\leq \frac{7n + 12}{10} \\[6pt]
&\leq \frac{3n - 3}{4} \\[6pt]
&\leq \left[ \frac{3n}{4} \right]
\end{align}
$$

---
