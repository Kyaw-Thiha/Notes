## Algorithm
```python
SELECT(A,k):
	if |A| < 39: then
		sort A;
		return A[k];
	else:
		s := CHOOSE SPLITTER(A)
		A- := sequence of elements of A < s
		A+ := sequence of elements of A > s
		
		if k <= | A- |, then return SELECT(A-,k);
		else if K=|A-| + 1, then return s;
		else, return SELECT(A+, k - | A- | + 1);
```

---
## Time Complexity
We can define the [[Time Complexity|time complexity]] as
$$
T(n) = T(\max(i-1), n-i) + cn
$$
where $s=i^{th}$ smallest.

### Best Case
$$
\begin{align}
T(n) &= T\left( \frac{n}{2} \right) + cn \\[6pt]
&= \Theta(n)
\end{align}
$$
where $a=1$, $b=2$, and $d=1$.

### Worst Case
$S$ is $\min$ or $\max$. This implies that
$$
\begin{align}
T(n) &= T(n-1) + cn \\[6pt]
&= c(n + (n-1) + (n-2) + \dots + 1) \\[6pt]
&= \Theta(n^{2})
\end{align}
$$

- Not necessary to shrink by factor
  Any $b>1$ will do.
- Note necessary to pick splitter that shrinks input size by factor of $b$ everytime.

---
## R-Select Algorithm
```python
CHOOSE SPLITTER(A)
r := RAND(1, |A|)   <- Each i in [1...n] picked with prob 1/n
return A[r]
```

[[Time Complexity|Expected running time]] is $\Theta(n)$.

---
In a `Good Splitters`,
- at least $\frac{1}{4}$ of elements are $< s$ and
- at least $\frac{1}{4}$ of elements are $> s$ 

![image|300](https://notes-media.kthiha.com/Selection-Algorithm/cc29e0ac9ab9a92008189beb52c07d64.png)

---
Random Variable $Z$ $= \# \text{ of trials until a good splitter is chosen}$.
$$
\begin{align}
E[Z]
&= \sum_{v} P[Z=v] \cdot v \\[6pt]
&= \frac{1}{2} \cdot 1 + \frac{1}{4} \cdot 2 + \frac{1}{8} \cdot_{3} + \dots 
\\[6pt]
&= \sum i\left( \frac{1}{2} \right)^{i} \\[6pt]
&= \frac{\frac{1}{2}}{\left( 1 - \frac{1}{2} \right)^{2}} \\[6pt]
&= \frac{\frac{1}{2}}{\frac{1}{4}} \\[6pt]
&= 2
\end{align}
$$

using 
$$
\sum_{i} ix^{i} = \frac{x}{(1-x)^{2}}
$$

---
- $n_{i} = \text{size of inupt in the } i^{th} \text{ recursive call}$
- $n_{0} = n$

![image|600](https://notes-media.kthiha.com/Selection-Algorithm/6ccb228d9846cd4d3956fe130aab2128.png)

Phase $j$ $= \text{ calls in which input size} \leq (\frac{3}{4})^{i} \cdot n$ and $> (\frac{3}{4})^{i+1} \cdot n$.

- Random Variable $\text{X} = \# \text{ of steps taken by algo}$
- Random Variable $\text{X}_{j} = \# \text{ of steps taken by algo during phase-j calls}$

$X = \sum_{j} x_{j}$
$$
X_{j} \leq 
\underbrace{
\begin{pmatrix}
\max \text{ \# of steps} \\ 
\text{in one phase j call}
\end{pmatrix}}_{c \ \left( \frac{3}{4} \right)^{j} \ n}
\cdot
\underbrace{
\begin{pmatrix}
\max \text{ \# of} \\ 
\text{phase j calls}
\end{pmatrix}
}_{Y_{j} \leftarrow \text{random variable}}
$$
We can then derive
$$
\begin{align}
\mathbb{E}[X_{j}] &\leq \mathbb{E}\left[ cn \ \left( \frac{3}{4} \right)^{j}  
\cdot Y_{j} \right] \\[6pt]
&= cn \ \left( \frac{3}{4} \right)^{j} \cdot \mathbb{E}[Y_{j}] \\[6pt]
&= 2cn \ \left( \frac{3}{4} \right)^{j}
\end{align}
$$
We then get
$$
\begin{align}
\mathbb{E}[X] &= \mathbb{E}\left[ \sum_{j} X_{j} \right] \\[6pt]
&= \sum_{j} \mathbb{E}[X_{j}] \\[6pt]
&\leq \sum_{j} 2cn \ \left( \frac{3}{4} \right)^{j} \\[6pt]
&= 2cn \ \sum_{j}\left( \frac{3}{4} \right)^{j} \\[6pt]
&= 8cn
\end{align}
$$

---
## Deterministic D-Select
