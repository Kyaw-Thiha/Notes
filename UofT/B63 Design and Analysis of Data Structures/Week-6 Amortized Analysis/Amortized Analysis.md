# Amortized Analysis
[[Amortized analysis]] is a worst-case analysis of a sequence of operations. 

![Amortized Analysis|200](https://media.licdn.com/dms/image/v2/D4E22AQHeQynsNCez5g/feedshare-shrink_1280/B4EZsisSVQJQAs-/0/1765813595642?e=1779321600&v=beta&t=hTvIiM0Y_IyIg9k881oKEgMM8489fMbgfuOzrVtj6Xs)

It is used to obtain a tighter bound on the [[Time Complexity|overall or average cost]] operation in the sequence than what is obtained by separately analyzing each operation.

---
## Properties
- [[Amortized analysis]] is an upper bound.
  It is the average performance of each operation in the worst case.
- [[Amortized analysis]] is concerned with the overall cost of a sequence of operations. It does not say anything about the cost of a specific operation in that sequence.
- [[Amortized analysis]] does not involve probability.
  This is how amortized analysis differs from the average-case time complexity of one operation.
---
## Amortized Sequence Complexity
The [[Amortized Analysis|amortized sequence complexity]] of a sequence of $m$ operations is:
$$
\text{amortized complexity}
= \frac{\text{worst-case seq complexity}}{m}
$$
The worst-case sequence [[Time Complexity|complexity]] of a sequence of $m$ operations is the max total time over all sequences of $m$ operations.

Therefore, the worst-case seq complexity is less than or equal to $m$ times the worst-case [[time complexity]] of a single operation in any seq of $m$ operations.

---
## Methods of Amortized Complexity
There are $3$ basic methods for computing [[Amortized Analysis|amortized complexity]]:
- [[Aggregate Method (Amortized Analysis)|Aggregate Method]]
- [[Accounting Method (Amortized Analysis)|Accounting/Banker's Method]]
- [[Potential Method (Amortized Analysis)|Potential/Physicist's Method]]

---
## See Also
- [[Amortized Analysis]]
- [[Aggregate Method (Amortized Analysis)]]
- [[Accounting Method (Amortized Analysis)]]
- [[Potential Method (Amortized Analysis)]]
- [[Dynamic Array]]
