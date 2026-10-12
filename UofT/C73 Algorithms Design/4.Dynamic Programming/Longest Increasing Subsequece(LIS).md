# Longest Increasing Subsequence(LIS)
[[Longest Increasing Subsequece(LIS)|Longest increasing subsequence]] is finding an increasing subset of numbers from left to right, that may not necessarily be immediately after each other. 

![300](https://assets.leetcode.com/users/images/5a09ce7b-2181-4627-b393-bcae69f465ab_1613480139.9677143.png)

---
## Example
Suppose we have the following sequence of
$$
A = 6,7,4,2,5,3,5,9,8,5
$$
Then, 
- $7,2,5,5,8$ is a subsequence of $A$
- $6,7,9$ is an increasing subsequence of $A$
- $2,3,5,9$ is a [[Longest Increasing Subsequece(LIS)|longest increasing subsequence]] of $A$

---
## Formal Definition
> **Input**: Sequence $A[1\dots n]$ of integers.
> **Output**: [[Longest Increasing Subsequece(LIS)|Longest increasing subsequence]] of $A$.

- $A$ is increasing if every element other than the first is greater than its predecessor.
- A [[Longest Increasing Subsequece(LIS)|subsequence]] of $A$ is a sequence consisting of elements of $A$, in the order in which they appear in $A$.
- Note that the elements of a subsequence of $A$ may not be consecutive in $A$.

---
## Dynamic Programming
### Naive Option
> Note that the brute force method of considering all [[Longest Increasing Subsequece(LIS)|subsequences]] of $A$ is inefficient since a sequence of length $n$ has $2^{n}$ subsequences.

### Dynamic Programming Problem
> So, we first try to express a supposed solution to the given instance of the problem as an extension of the solution to a smaller instance of a problem.

> Suppose we have found a [[Longest Increasing Subsequece(LIS)|LIS]] $S$ of $A$. $S$ must end in some position of $A$, say $i$.
> We can rewrite $S$ as concatenation of smaller [[Longest Increasing Subsequece(LIS)|LIS]] $S'$ and another symbol $A[i]$.

$$
S=S'A[i]
$$
where $S'$ is a 
- $(a)$ longest
- $(b)$ increasing 
- $(c)$ subsequence of $A$ that ends in position $j$, for some $j<i$ such that $A[j] < A[i]$.

We can prove that $S'$ is longest, increasing through a `cut-and-paste argument`.

---
## Proof
> **What we already know**
> $S'$ must satisfies $(b)$ and $(c)$ since if $S'$ isn't increasing or didn't end at a smaller position before $i$, then $S'A[i]$ wouldn't be an increasing subsequence.
> 
> **Claim**
> $S'$ is the longest subsequence of $A$ $\big(\text{property } (a) \big)$ with properties $(b)$ and $(c)$.

### Cut-And-Paste Argument
> For sake of **contradiction**, suppose $S'$ is not the longest. 
> Then, there is some $S''$ with properties $(b)$ and $(c)$ that is longer than $S'$.

> Since $S''$ is increasing and ends at position $j < i$ with $A[j] <A[i]$, appending $A[i]$ gives a valid increasing subsequence $S''A[i]$.

> But then $S''A[i]$ is an increasing subsequence of $A$ that is longer than $S'A[i] = S$.
> This contradicts that $S$ is a [[Longest Increasing Subsequece(LIS)|LIS]] of $A$.

---
### Cut-And-Paste Argument Structure
> Take an **optimal solution** and decompose it into a smaller piece.
> 
> Suppose the **smaller piece is not optimal** for its own subproblem. 
> So there’s a better one.

> **Cut** the smaller piece out and **paste** the better one in its place.
> 
> Check that the **result is still valid**. The better piece must be compatible with the rest of the solution. This may require real work.

> Show the **new solution beats the original.**
> 
> **Contradiction.** The original was supposed to be optimal. 
> So the smaller piece must have been optimal after all.

---
## Optimal Substructure Property
> **Optimal Substructure Property**
> Every optimal solution to an instance of the problem extends an optimal solution to a smaller instance of the problem.

This [[#Dynamic Programming|formulation]] shows that [[Longest Increasing Subsequece(LIS)|LIS]] exhibit `optimal substructure property`.
$$
\underbrace{S}_{\text{LIS of A ending at i}} = \underbrace{S'}_{\text{LIS of A ending at j<i with } a_{j} < a_{i}} a_{i}
$$

### Example
Another example is finding shortest paths: 
$$
\underbrace{u_{1}, u_{2}, u_{3}, \dots, u_{k-1}}, u_{k}
$$
A counterexample of this is finding longest paths.

---
### Subproblems
We can see how we can retrofit the algorithm that computes the length of [[Longest Increasing Subsequece(LIS)|LIS]] to also compute an actual [[Longest Increasing Subsequece(LIS)|LIS]].

### Length of LIS
Compute the following:
$$
L(i) = \text{length of LIS of A} \text{(that ends in position i)}^{(*)}
$$
Note that the constraint $(^{*})$ about ending at position $i$ is guided by our [[#Optimal Substructure Property|optimal substructure property]]. This gives the [[Longest Increasing Subsequece(LIS)|opitmal LIS]] that ends in some position in terms of [[Longest Increasing Subsequece(LIS)|optimal LIS]] that ends in earlier position.

> We can easily compute the quantity we want by taking $\max\{ L(i): i \in [1\dots n] \}$.

---
### Recursive Formula
In terms of $S = S'A[i]$, we have that $L(i) = 1 + L(i)$ for some $j < i$ such that $A[j] < A[i]$. 
But we don't know which $j$ so we try out all possibilities and pick the best.

> Since we want to maximize $L(i)$, we pick a $j<i$ such that $A[j] < A[i]$ that has maximum $L(j)$. 

We shall thus define as follows:
$$
L(i) = \begin{cases}
\max\{ L(j): j<i \text{ and } A[j] < A[i] \} + 1 
&, \quad \text{if } \exists j \ s.t. \ j<i \ \& \ A[j] < A[i]  \\[6pt]
1 & , \quad \text{otherwise}
\end{cases}
$$
or equivalently as
$$
L(i) = \begin{cases}
1 & , \quad \forall j \in [1\dots i-1], A[j] \geq A[i] \\[6pt]
1 + \max\{ L(j): j<i \text{ and } A[j] < A[i] \} & ,  
\quad \text{otherwise} \quad  (\dagger)
\end{cases}
$$
In other words,

> For all $j \in [1\dots i-1], \ A[j]\geq A[i]$.
> Then, the longest increasing subsequence of $A$ that ends in position $i$ consists of $A[i]$ alone, and so $L(i) = 1$.

> For some $j \in [1 \dots i-1], \ A[j] < A[i]$.
> Then, the [[Longest Increasing Subsequece(LIS)|longest increasing subsequence]] of $A$ that ends in position $i$ is $S = S'A[i]$.
> Thus, $L(i) = 1 + \max\{ L(j): j<i \text{ and } A[j] < A[i] \}$.

---
## Pseudocode
### Finding Length
```python
LIS(A[1..n]):
	m := 1
	for i:=1 to n do:
		L(i) := 1;
		
		# Looking at all predecessors of L(i)
		for j:=1 to i-1 do:
			if A[j] < A[i] and L(i) < L(j) + 1:
				then L(i) := L(j) + 1;
		# Updating m 
		if L(i) > L(m):
			then m:=i
	return L(m)
```

---
### Finding the Sequence(Variant-1)
In here, we compute a new value to get us the sequence.

```python
LIS(A[1..n]):
	m := 1
	for i:=1 to n do:
		L(i) := 1; 
		predecessor(i) := 0;
		
		# Looking at all predecessors of L(i)
		for j:=1 to i-1 do:
			if A[j] < A[i] and L(i) < L(j) + 1:
				L(i) := L(j) + 1; 
				predecessor(i) := j;
		# Updating m 
		if L(i) > L(m):
			then m:=i
	# Returning the Sequence
	S := empty seq
	repeat
		prepend A[m] to S
		m := predecessor(m)
	until m = 0
	return S
```

---
### Finding the Sequence(Variant-2)
In here, we use existing values derivation to get the sequence we want.

```python
i := m; 
S := empty seq;
while i != 0: do
	prepend A[i] to S
	j := i-1
	while j>=1 and (L(j) != L(i)-1 or A[j]>=A[i]): do
		j:=j-1
	i:=j
return S
```

### Running Time
The [[Time Complexity|running time]] is of $\Theta(n^{2})$.

---
