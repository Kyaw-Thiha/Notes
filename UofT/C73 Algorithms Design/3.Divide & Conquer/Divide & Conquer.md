# Divide & Conquer
```python
Mergesort(A[1...n]):
	if n=1, done
	else split A into A1, A2 each of size n/2
	recursively sort A1 & A2
	merge two sorted arrays
```

In here $b=a=2$.

---
## Binary Search
```python
BinarySearch(A[1...n], x):
	if n=1, trivial
	else if A[n/2] >= x then
		recursively search A[1...[n/2]]
	else
		recursively search A[[n/2+1]...n]
```

In here, $b=2$ and $a=1$(since we only continue in one half).

---
## Generic Divide & Conquer
```python
if n is "small", solve directly
else
	- split input into a pieces, each of size (n/b) where b>1
	- recursively solve each subproblems
	- combine solutions of subproblems into solution of given problem
```

---
## Runtime Complexity
### Merge Sort
$$
T(n) = 2\cdot T\left( \frac{n}{2} \right) + c \cdot n
$$

### Binary Search
$$
T(n) = T\left( \frac{n}{2} \right) + c
$$

### Generic Divide & Conquer
$$
\boxed{ \ 
T(n) = a \cdot T\left( \frac{n}{b} \right) + c \cdot n^{d}
\ }
$$
where $n$ is divisible by $b$, and $b$ is an integer.

We could generalize it to
$$
\boxed{ \ 
T(n) = a_{1} \ T\left( \left[ \frac{n}{2} \right] \right)
+ a_{2} \ T\left( \left[ \frac{n}{b} \right] \right)
+ cn^{d}
\ }
$$
where $a_{1} + a_{2} = a$.

---
## Master Theorem
Suppose $T(n)$ is defined as
$$
T(n) = a_{1} \ T\left( \left[ \frac{n}{2} \right] \right)
+ a_{2} \ T\left( \left[ \frac{n}{b} \right] \right)
+ cn^{d}
$$
(a): If $a < b^{d}$, then $T(n) = \Theta(n^{d})$.
(b): If $a=b^{d}$, then $T(n) = \Theta(n^{d} \log n)$
(c): If $a > b^{d}$, then $T(n) = \Theta(n^{\log_{b}(a)})$

---