# Amortized Analysis Tips

## Aggregate Method
- Computes the worst-case complexity. 
- Then divide by no. of operations.

A unique case is [[Dynamic Array]].

[[Aggregate Method (Amortized Analysis)]]

---
## Accounting Method
- Take the time complexity of each operation as real cost.
- Define artificial costs $\hat{c}_{i}$ of each operation $i$ such that 
	- "push" operations accumulate charge to cover expensive operation
	- expensive operation/case use up the charges
	- "pop" operation in simple case cover their own cost
- Total time complexity is 

$$
\text{Total Complexity} = \sum_{i} n \ \hat{c}_{i}
$$
- Divide by $n$ to get amortized analysis.

To prove that
- Let amount = total stored credit. 
- Defines an invariant of 

$$
\text{amount} \geq \sum_{i} c \cdot |S_{i}|
$$
where $|S_{i}|$ is the length of certain data structure.
- Prove it for each of the cases

If pop operation cover its own cost, we only need one $|S_{i}|$.

[[Accounting Method (Amortized Analysis)]]

---
## Potential Method
- Define the potential function $\Phi(D_{i})$.
  Link it to the variable that defines the potential charge like the length of data structure $|S_{j}|$.

$$
\Phi(D_{i}) = c \cdot |S_{j}|
$$
- Analyze each operation and cases with 

$$
a_{i} = t_{i} + \Phi(D_{i}) - \Phi(D_{i-1})
$$

[[Potential Method (Amortized Analysis)]]

---