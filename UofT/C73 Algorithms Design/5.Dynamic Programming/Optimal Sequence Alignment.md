# Optimal Sequence Alignment (Edit Distance)
[[Optimal Sequence Alignment|Optimal sequence alignment]] can get us the similarity between two strings in terms of the number of edits needed to transform one string into another.

![300](https://www2.seas.gwu.edu/~simhaweb/cs151/lectures/module12/figures/align5.gif)

---
## Formal Definition
Given two strings, we can define a measure of similarity between them as having minimum number of edits. 

> **Input**: Strings: $x[1\dots m]$ and $y[1\dots n]$.
> **Output**: Edit distance between $x$ and $y$.

We then have the following functions: `insert()`, `delete()`, `change()`.

### Alignment
An [[Optimal Sequence Alignment|alignment]] of strings $x$ and $y$ is a pair of strings $x'$ and $y'$ so that
- $(a)$ $x'$ is $x$ possibly with some $\text{"-"}$ symbols interspersed and the similarly for $y'$
- $(b)$ $|x'| = |y'|$
- $(c)$ there is no position $i$ so that $x'[i] = y'[i] = -$.

### Definitions
> **Cost**
> The `cost` of an alignment of $x$ and $y$ is optimal if there is no alignment of $x$ and $y$ with lower cost. 

> **Edit Distance**
> The `edit distance` between $x$ and $y$ is the cost of an optimal alignment.

> **Invertibility**
> Since the operations `insert()` and `delete()` are inverses of each other, the [[Optimal Sequence Alignment|edit distance]] between $x$ and $y$ is the same as the [[Optimal Sequence Alignment|edit distance]] between $y$ and $x$.

---
## Example
Consider the following example:
![image|300](https://notes-media.kthiha.com/Optimal-Sequence-Alignment/a7ce9f7bb168f29c05ea41ac2abef6f9.png)

Alternatively, we could do 

![image|300](https://notes-media.kthiha.com/Optimal-Sequence-Alignment/20539d1517f25701ab739ba3ed7112cc.png)

---
## Solution
Consider [[Optimal Sequence Alignment|optimal alignment]] $X, Y$ of $x, y$.

![image|500](https://notes-media.kthiha.com/Optimal-Sequence-Alignment/49a99b6949dfccf1c9cf254862236d78.png)

Then, we have $3$ cases to consider.

---
### Subproblems to Solve: Compute
$$
ED(i,j) = \text{edit distance between } x[1\dots i] \ \& \ y[1\dots j]
, \ \quad \forall i \in [0\dots m], \forall j \in [0\dots n] \quad (*)
$$
The recursive formula to compute $(*)$ is
$$
ED(i,j) = \begin{cases}
i & , \ \text{if } j=0 \\[6pt]
j & , \ \text{if } i=0 \\[6pt]
\min(\text{ED}(i-1, j-1) + \delta(i,j), \ \text{ED}(i-1,\ j)+1,  
\ \text{ED}(i, \ j-1) + 1) & , \ \text{o.w} \ (\dagger) \\[6pt]
\end{cases}
$$
which we shall represent as $(\dagger)$.

---
## Pseudocode
```python
EditDist(x[1...m], y[1...n]):
	for i:=0 to m, do:
		ED(i,0):=i
	for j:=1 to n, do:
		ED(0,j):=j
	for i:=1 to m, do:
		for j:=1 to n, do:
			ED(i,j):=(†)
	
	return ED(m,n)
```

### Running Time
The [[Time Complexity|running time]] of this algorithm is $\Theta (m, \ n)$.

---
