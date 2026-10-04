# Longest Increasing Subsequence(LIS)
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
## Problem Formulation
- **Input**: Sequence of numbers $A$
- **Output**: [[Longest Increasing Subsequece(LIS)|Longest increasing subsequence]] of $A$

Let $S=$ [[Longest Increasing Subsequece(LIS)|LIS]] of $A$. Then, let 
$$
S=S'a_{i}
$$
Then, $S'$ is a 
- longest
- increasing 
- subsequence of $A$ that ends at position $j < i$ & $a_{j} < a_{i}$

We can prove that $S'$ is longest, increasing through a `cut-and-paste argument`.

---
## Optimal Substructure Property
$$
\underbrace{S}_{\text{LIS of A ending at i}} = \underbrace{S'}_{\text{LIS of A ending at j<i with } a_{j} < a_{i}} a_{i}
$$

Another example is finding shortest paths: 
$$
\underbrace{u_{1}, u_{2}, u_{3}, \dots, u_{k-1}}, u_{k}
$$
A counterexample of this is finding longest paths.

---
## Subproblems
### Length of LIS
Compute the following:
$$
L(i) = \text{lenght of LIS of A ending at i}
$$
We shall first define as following:
$$
L(i) = \begin{cases}
\max\{ L(j): j<i \text{ and } a_{j} < a_{i} \} + 1 
&, \quad \text{if } \exists j \ s.t. \ j<i \ \& \ a_{j} < a_{i}  \\[6pt]
1 & , \quad \text{otherwise}
\end{cases}
$$

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
