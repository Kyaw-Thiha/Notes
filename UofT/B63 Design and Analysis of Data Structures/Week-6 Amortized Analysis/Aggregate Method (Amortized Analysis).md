# Aggregate Method
Computes the worst-case seq [[Time Complexity|complexity]] of a seq of operations, and divides by the number of operations in the seq.

---
## Example
Suppose we have an [[Augmented Data Structures|augmented stack]] with the operations:
- $\text{push}(s,x): \theta(1)$
- $\text{pop}(s): \theta(1)$
- $\text{multiPop}(s,k):$ Removes the top $k$ elements from the stack. $\theta(k)$

Consider performing a total of $n$ operations from among $\text{push()}, \ \text{pop()},$ and $\text{multiPop}()$ on a stack that is initially empty.

- The stack can contain at most $n$ elements.
- Furthermore, we know that the cost of each operation for $\text{multiPop()}$ is $O(n)$.
- Therefore, the total cost is $O(n^{2})$.
- However, we can use the [[Aggregate Method (Amortized Analysis)|aggregate method]] to get a much tighter upper bound. We know that we can only pop an element if we've pushed the element first.
- Since there can be at most $n$ pushes, there can be at most $n$ pops, including counting the appropriate number of pops for multipop.
- This implies that the total time taken for the entire sequence is at most $O(n)$.
- This gives us that each operation takes an average of $O(1)$.

> **Note**:
> The [[Aggregate Method (Amortized Analysis)|aggregate method]] applies the cost to each operation, even when there are several type of operations in the sequence.

---
## See Also
- [[Amortized Analysis]]
- [[Aggregate Method (Amortized Analysis)]]
- [[Accounting Method (Amortized Analysis)]]
- [[Potential Method (Amortized Analysis)]]
- [[Dynamic Array]]

