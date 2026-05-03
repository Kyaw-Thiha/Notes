# Potential/Physicist's Method
Suppose we can define a function $\phi$ of a data structure with the following properties:
- $\phi(h_{0}) = 0$, where $h_{0}$ is the initial state of the data structure.
  $\text{I.e:}$ Our data structure starts with no potential.
- $\phi(h_{t}) > 0$ for all states $h_{t}$ of the data structure occuring during the course of the computation.
- This means that there is never negative potential.

---
## Potential Function
The [[Potential Method (Amortized Analysis)|potential function]] $\phi$ is designed to keep track of the pre-charged [[Time Complexity|time or cost]] at any point in the computation.
- It measures how much saved-up time is available to pay for expensive operations.
- This is analogous to the bank credit in the [[Accounting Method (Amortized Analysis)|accounting method]].
- The difference is that with the [[Potential Method (Amortized Analysis)|potential method]], it depends only on the current state of the data structure, and not the history of the computation that got it in that state.


---
## Amortized Time
We define the [[Amortized Analysis|amortized time]] $T_{A}(i)$ of operation $i$ as the actual time plus the change in potential.
$$
T_{A}(i)
= C_{i} + \underbrace{\phi(h_{i+1}) - \phi(h_{i})}_{\text{Change in Potential}}
$$
where
- $C_{i}$ is the actual [[Time Complexity|cost]] of the operation.
- $h_{i}$ is the state of the data structure before the operation.
- $h_{i+1}$ is the state of the data structure after the operation.

> **Note on Potential Function**
> Ideally, the [[Potential Method (Amortized Analysis)|potential function]] $\phi$ should be defined so that the [[Potential Method (Amortized Analysis)|amortized time]] of each operation is small.
> Because of this, the change in potential should be positive for low cost operations, and negative for high cost operations.

> **Note on Amortized Time**
> For the [[Potential Method (Amortized Analysis)|potential method]] to be valid, we need the [[Potential Method (Amortized Analysis)|amortized time]] to be an over-estimate of the actual cost.

---
## Proof of Potential Method
**Claim**: The total amortized cost is the upper bound on the total actual cost.

Total amortized time is sum of the individual amortized times. Let's sum up the amortized cost for a seq of $n$ operations.
$$
\begin{align}
\sum^{n-1}_{i=0} T_{A}(i)
&= \sum^{n-1}_{i=0} (C_{i} + \phi(h_{i+1})  
- \phi(h_{i})) \\[6pt]
&= \sum^{n-1}_{i=0} C_{i} + \sum ^{n-1}_{i=1}  
\phi(h_{i+1}) - \sum ^{n-1}_{i=0} \phi(h_{i}) 
\\[6pt]
&= \sum^{n-1}_{i=0} C_{i} 
+ \underbrace{\sum^{n}_{i=1} \phi(h_{i})  
- \sum ^{n-1}_{i=0} \phi(h_{i})}_{\text{Telescoping Sum}} 
 \\[6pt]
&= \sum ^{n-1}_{i=0} C_{i} + \phi(h_{n}) - \phi(h_{0}) \\[6pt]
&= \text{Total Cost} + \phi(h_{n})
\end{align}
$$

---
## Example 
Consider a [[dynamic array]].

We need $\phi$ to depend on the current state of the array.
This means that
- the more full the array is, the higher the [[Potential Method (Amortized Analysis)|potential]] should be.
- and the more empty the array is, the more time there is to build [[Potential Method (Amortized Analysis)|potential]].

We need the number of items in the array $n$ to play a role.
- When the array doubles, we need the current potential to decrease. 
- In fact, when the array doubles, the potential should be $0$ or close to $0$.

**Getting the potential function**:
If we let $m$ be the size of the array, then we know that $m=2n$ when the array doubles.
From this, we can get the function
$$
\phi(h) = 2n - m
$$

**Checking the potential function**
Let's check if $\phi(h)$ satisfies all the requirements:
- An array of length $0$:
	- We want $\phi(h_{0}) = 0$.
	- Since the array is of length $0$, $n$ and $m$ are $0$.
	  $\therefore$ $2n - m = 0$, as wanted.
	- Once we add an element, we have an array of size $1$ and we always double the array when its full.
	  This means that $2n \geq m$ at all times, so $\phi(h_{t}) \geq 0$.
- What's the cost of appending an element?
- **Case-1**: $n < m$
	- The actual cost is $1$.
	- Furthermore, $n$ increases by $1$ and $m$ doesn't change. 
	  So, the [[Potential Method (Amortized Analysis)|potential]] increases by $2$.
	- $\therefore$ The amortized time is $1+2=3$.
- **Case-2**: $n=m$
	- The array is doubled so the actual cost is $n+1$.
	- The potential before the double is
$$
\begin{align}
2n-m
&= 2n-n \\
&= n
\end{align}
$$
	- The potential after the double is 
$$
\begin{align}
2(n+1) - 2(n) = 2
\end{align}
$$
	- $\therefore$ The amortized time is $1+2 = 3$.

---
## See Also
- [[Amortized Analysis]]
- [[Aggregate Method (Amortized Analysis)]]
- [[Accounting Method (Amortized Analysis)]]
- [[Potential Method (Amortized Analysis)]]
- [[Dynamic Array]]
