# Selection Algorithm
In [[Selection Algorithm|order statistics]], we want to find the maximum, minimum and the median of numbers.

![300](https://ajitvoice.wordpress.com/wp-content/uploads/2020/03/images12887284019204913391.jpg?w=723)

Maximum and minimum can be found in $\Theta(n)$, but for median the brute force way costs $\Theta(n \log n)$.
The [[Selection Algorithm|selection algorithm]] aims to find the median in $\Theta(n)$ too.

---
## Formal Definition
> **Input**: An array $A[1\dots n]$ of distinct numbers(in any order), and an integer $k \in [1\dots n]$.
> **Output**: The $k^{th}$ smallest element in $A$.

---
## Algorithm
### Idea
> **Less Than Threshold**
> If $len(A) \leq \text{threshold(here its 39)}$, then we sort them normally with $\Theta(n \log n)$.

> **Greater Than Threshold**
> If $len(A) > \text{threshold}$, we first choose a splitter to get $\text{A-} < s$, $\text{A+} > s$ and splitter $s$.

![image|400](https://notes-media.kthiha.com/Order-Statistics/f2438333fce3ff7aac9b07ba257260fc.png)

> **Recursive Conditions**
> Recall that $k$ is the $k^{th}$ smallest element to be found.
> $(a)$ If $k \leq |\text{A-}|,$ we do the recursive call for $\text{A-}$ at $k$.
> $(b)$ If $k = |\text{A-}| + 1,$ the splitter is the $k^{th}$ smallest element.
> $(c)$ If $k > |\text{A-}| + 1,$ we do recursive call for $\text{A+}$ at the next index $\big( k - (|\text{A-}| + 1) \big)$.

---
### Food-for-Thought
> **Generalizing**
> Sometimes to prove a statement by induction it pays to **generalize** it and prove a **stronger** statement. The same logic applies the idea behind this algorithm.

---
### Pseudocode
```python
SELECT(A, k):
	if |A| < 39: then
		sort A;
		return A[k];
	else:
		s := CHOOSE_SPLITTER(A)
		A- := sequence of elements of A < s
		A+ := sequence of elements of A > s
		
		if k <= | A- |: 
			then return SELECT(A-, k);
		else if k = | A- | + 1: 
			then return s;
		else: 
			return SELECT(A+, k - | A- | - 1);
```

---
## Time Complexity
We can define the [[Time Complexity|time complexity]] as
$$
T(n) = T(\max(i-1), n-i) + cn
$$
where $s=i^{th}$ smallest.

---
### Best Case
We get the best case when the [[#Splitters|splitter]] leads to the larger of $\text{A-}$ and $\text{A+}$ being $\frac{n}{b}$ of $n$.
$$
\begin{align}
T(n) &= T\left( \frac{n}{2} \right) + cn \\[6pt]
&= \Theta(n)
\end{align}
$$
where $a=1$, $b=2$, and $d=1$.

Note  the following
- Not necessary to shrink by any specific factor. Any $b>1$ will do.
- Not necessary to pick splitter that shrinks input size by factor of $b$ every time.
  Even if it shrinks by $b$ every $k$ times, we still have $\Theta(kn) = \Theta(n)$.

---
### Worst Case
We get the worst case when $s$ is minimum or maximum element of array.
$$
\begin{align}
T(n) &= T(n-1) + cn \\[6pt]
&= c(n + (n-1) + (n-2) + \dots + 1) \\[6pt]
&= \Theta(n^{2})
\end{align}
$$


---
## Splitters
### R- Select
> Randomly select the median in a given set.
> **Worst Case Time Complexity**: $\Theta(n^{2})$
> **Expected Time Complexity**: $\Theta(n)$
> 
> [[Random Select (R-Select)|Read More]]

---
### D-Select
> Group partitions into $5$ elements each. 
> Find median of each those groups. 
> Recursively call the `Select(A, k)`.
> **Worst Case Time Complexity**: $\Theta(n)$
> 
> [[Deterministic Select (D-Select)|Read More]]

---