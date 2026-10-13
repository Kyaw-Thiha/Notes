# Divide & Conquer

![500](https://favtutor.com/resources/images/uploads/Divide_and_Conquer_Algorithm.png)

In [[Divide & Conquer|divide & conquer algorithms]], 
- the algorithm actively choose how to divide its input into subproblems
- number of subproblems is fixed
- size of each subproblem is fixed fraction of the original


---
## Generic Algorithms
```python
if n is "small", solve directly
else
	- split input into a pieces, each of size (n/b) where b>1
	- recursively solve each subproblems
	- combine solutions of subproblems into solution of given problem
```

---
### Merge Sort
```python
Mergesort(A[1...n]):
	if n=1, done
	else split A into A1, A2 each of size n/2
	recursively sort A1 & A2
	merge two sorted arrays
```

In here $b=a=2$.

---
### Binary Search
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
## Algorithms
### Karatsuba Binary Multiplication
Takes advantage of 
$$
X \cdot Y  = X_{1}Y_{1}2^{n} + (X_{1}Y_{0} + X_{0}Y_{1}) \cdot 2^{n} + X_{0}Y_{0}
$$
or even
$$
X \cdot Y = X_{1}Y_{1}2^{n} + ((X_{1} + X_{0})  \cdot (Y_{1} + Y_{0})
- X_{1}Y_{1} - X_{0}Y_{0}) \cdot 2^{n/2} + X_{0}Y_{0}
$$
to get faster multiplication algoritm.

[[Karatsuba Binary Multiplication|Read More]]

---
### Closest Pair
[[Closest Pair of Points|Closest pair of points]] aim to find the pair of points that is closer than any other points in the space.

![300](https://media.geeksforgeeks.org/wp-content/uploads/20240712111230/Closest-Pair-of-Points-between-two-points.png)

---
### Selection Algorithm (Order Statistics)
Find the $k^{th}$ median of a list of numbers.

![300](https://ajitvoice.wordpress.com/wp-content/uploads/2020/03/images12887284019204913391.jpg?w=723)

[[Selection Algorithm|Read More]]

---
## Runtime Complexity
[[Time Complexity|Time complexity]] $T(n)$ can be described in 
$$
T(n) = \begin{cases}
a \cdot T\left( \frac{n}{b} \right) + c \cdot n^{d}
& \text{if } n > 1 \\[6pt]
c & \text{if } n=1
\end{cases}
$$
where
- $c\geq1$ and $d \geq 0$ are constants.
- the **first term** $a \cdot T\left( \frac{n}{b} \right)$ accounts for the time required to solve recursively the $a$ subproblems, each of size $\frac{n}{b}$.
- the **second term** $cn^{d}$ accounts for the time required to 
	- divide up the given input into smaller inputs.
	- and to compute the solution to the original instance from the solutions to smaller instances.

where the constants are
- $a:$ no. of `recursive subproblems` created
- $b:$ how much `smaller` each subproblem gets
- $c:$ the `constant multiplier` on the divide/combine work
- $d:$ `exponent` describing how divide/combine grows with input size $n$
	- $d=0:$ divide/combine work is constant (eg: [[AVL Tree|binary search]])
	- $d=1:$ divide/combine work is linear (eg: [[Divide & Conquer|merge sort]])
	- $d=2:$ divide/combine work is quadratic (eg: an algorithm that compares every pair of elements when combining)

---
### Merge Sort
For [[#Merge Sort|merge sort]], we have 
$$
T(n) = 2\cdot T\left( \frac{n}{2} \right) + c \cdot n
$$
where
- $a=2$ since we get twice as many subproblems each time.
- $b=2$ since we are dividing into $2$ subproblems each time.
- $d=1$ since in linear time, we can 
	- split $A$ into $A_{1}$ and $A_{2}$.
	- and merge the sorted versions of $A_{1}$ and $A_{2}$ into the final answer.

---

### Binary Search
For [[AVL Tree|binary search]], we have
$$
T(n) = T\left( \frac{n}{2} \right) + c
$$
where
- $a=1$ since the no. of subproblems remain the same.
- $b=2$ since the subproblems are divided by $2$ each time.
- $d=0$ because splitting and combining steps can be done in constant time.

---
### Generic Divide & Conquer
$$
\boxed{ \ 
T(n) = a \cdot T\left( \frac{n}{b} \right) + c \cdot n^{d}
\ }
$$
where $n$ is divisible by $b$, and $b$ is an integer.

---
### Generalized Divide & Conquer
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
The [[#Master Theorem|master theorem]] states that for $T(n) = a \cdot T\left( \frac{n}{b} \right) + c \cdot n^{d}$, we have
$$
T(n) = \begin{cases}
\Theta(n^{d}) & \text{if } a < b^{d} \\[6pt]
\Theta(n^{d} \ \log n) & \text{if } a=b^{d} \\[6pt]
\Theta(n^{\log_{b} (a)}) & \text{if } a > b^{d}
\end{cases}
$$
with
- $(a)$ If $a < b^{d}$, then $T(n) = \Theta(n^{d})$
- $(b)$ If $a=b^{d}$, then $T(n) = \Theta(n^{d} \log n)$
- $(c)$ If $a > b^{d}$, then $T(n) = \Theta(n^{\log_{b}(a)})$

> In the **first case**, the running time is dominated by time splitting the input and recombining the results of the recursive calls.
> 
> In the **second case**, the running time is dominated by the number of recursive calls made.

[[Master Theorem|Read More]]

---