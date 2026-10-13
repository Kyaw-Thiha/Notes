# Random Select
## Algorithm
```python
CHOOSE_SPLITTER(A):
	r := RAND(1, |A|)   <- Each i in [1...n] picked with prob 1/n
	return A[r]
```

[[Time Complexity|Expected running time]] is $\Theta(n)$.

---
## Running Time Proof
### Note
In a `Good Splitters`,
- at least $\frac{1}{4}$ of elements are $< s$ and
- at least $\frac{1}{4}$ of elements are $> s$ 

![image|300](https://notes-media.kthiha.com/Selection-Algorithm/cc29e0ac9ab9a92008189beb52c07d64.png)

---
### Expectation of getting a Good Splitter
Let random variable $Z$ $= \text{no. of trials until a good splitter is chosen}$.
$$
\begin{align}
E[Z]
= \sum_{v} P[Z=v] \cdot v 
= \frac{1}{2} \cdot 1 + \frac{1}{4} \cdot 2 + \frac{1}{8} \cdot 3 + \dots 
= \sum i\left( \frac{1}{2} \right)^{i} 
= \frac{\frac{1}{2}}{\left( 1 - \frac{1}{2} \right)^{2}} 
= \frac{\frac{1}{2}}{\frac{1}{4}} 
= 2
\end{align}
$$

by using 
$$
\sum_{i} ix^{i} = \frac{x}{(1-x)^{2}}
$$

---
### When a Good Splitter is chosen
Let each phase represent the elements from which a good splitter is chosen to the next.
- $n_{i} = \text{size of input in the } i^{th} \text{ recursive call}$
- $n_{0} = n$

![image|600](https://notes-media.kthiha.com/Selection-Algorithm/6ccb228d9846cd4d3956fe130aab2128.png)

Phase $j$ $= \text{ calls in which input size} \leq (\frac{3}{4})^{i} \cdot n$ and $> (\frac{3}{4})^{i+1} \cdot n$.

- Random Variable $\text{X} = \# \text{ of steps taken by algo}$
- Random Variable $\text{X}_{j} = \# \text{ of steps taken by algo during phase-j calls}$

So, $X = \sum_{j} X_{j}$.
$$
X_{j} \leq 
\underbrace{
\begin{pmatrix}
\max \text{ \# of steps} \\ 
\text{in one phase j call}
\end{pmatrix}}_{c \ \left( \frac{3}{4} \right)^{j} \ n}
\times
\underbrace{
\begin{pmatrix}
\max \text{ \# of} \\ 
\text{phase j calls}
\end{pmatrix}
}_{Y_{j} \leftarrow \text{random variable}}
$$
---
### Expectation over each phase
We can then derive
$$
\begin{align}
\mathbb{E}[X_{j}] &\leq \mathbb{E}\left[ cn \ \left( \frac{3}{4} \right)^{j}  
\cdot Y_{j} \right] \\[6pt]
&= cn \ \left( \frac{3}{4} \right)^{j} \cdot \mathbb{E}[Y_{j}] \\[6pt]
&= 2cn \ \left( \frac{3}{4} \right)^{j}
\end{align}
$$
---
### Expected Time Complexity
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
