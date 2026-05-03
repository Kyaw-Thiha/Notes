# Accounting/Banker's Method
In the [[Accounting Method (Amortized Analysis)|accounting method]], we do the [[Amortized Analysis|analysis]] as if we were an intermediate service providing access to the data structure.
- The cost to us for each operation is the operation's worst-case running time.
- We charge the customer for each operation such that we cover our costs with what we earn in charges.
- We aim for a total charge as close as possible to the total cost. This will give us the best estimate of the [[time complexity]].

The charge is the approximate [[Amortized Analysis|amortized complexity]] per operation.
- Typically, we will charge more for some types of operations and nothing for other types.
- When we charge more than the cost, the leftover amount can be stored with the elements in the data structure as credit.
- When we perform a free operation on an element, we can use the credit stored with that element to pay for the cost of the operation.

---
## Main Ideas
- In the [[Accounting Method (Amortized Analysis)|accounting method]], we assign differing charges to different operations, with some operations charged more or less than they actually cost.
- **Amortized Cost**: Amount we charge an operation.
- **Credit**: Used when an operation's amortized cost exceeds its actual cost. Assigned to specific objects in data structure.
- Credit can help pay for later operations whose amortized cost is less than the actual cost.
- Thus, we can view the amortized cost of an operation as being split between its actual cost and the credit that is either deposited or used up.
- Different operations may have different amortized costs.
- This method differs from [[Aggregate Method (Amortized Analysis)|aggregate analysis]], in which all operations have the same amortized cost.

---
### Additional Notes
- We must choose the amortized costs of operations carefully.
  We want to assign charges and distribute credits carefully $s.t.$ 
	- we can ensure that each operation's costs will be payed 
	- and that the total credit stored in the data structure is never negative.
- The total amount charged for a sequence of operations is an upper bound on the total cost of the sequence.
  This means that we can use the total charge to compute an upper bound on the [[Amortized Analysis|amortized complexity]] of the sequence.

---
## Example
Suppose we have an [[Augmented Data Structures|augmented stack]] with the operations:
- $\text{push}(s,x): \theta(1)$
- $\text{pop}(s): \theta(1)$
- $\text{multiPop}(s,k):$ Removes the top $k$ elements from the stack. $\theta(k)$

Hence, the cost of each operation (representing the [[time complexity]] of each operation) is as follows:
- $\text{cost}(\text{push}(s,x)) = 1$
- $\text{cost}(\text{pop}(s)) = 1$
- $\text{cost}(\text{multiPop}(s,k)) = \min(k, |s|)$

However, each element can take part in at most two operations: one push and one pop/multipop.
Therefore, the total cost for one element is $2$.

We can assign charges like this:
- $\text{charge}(\text{push}) = 2$
- $\text{charge}(\text{pop}) = 0$
- $\text{charge}(\text{multiPop}) = 0$

This way, every time we push an element, we will have enough credit to $\text{pop}$/$\text{multiPop}$ it.

The total charge for $m$ operations is at most $2m$, so the total cost is $O(m)$. To get the [[Amortized Analysis|amortized complexity]], we divide the total cost by the number of operations.

In this case, dividing $O(m)$ by $m$ gets us an [[Amortized Analysis|amortized complexity]] of $O(1)$ per operation.

---
## Comparism to Aggregate Method
One advantage the [[Accounting Method (Amortized Analysis)|accounting method]] has over the [[Aggregate Method (Amortized Analysis)|aggregate method]] is that different operations can be assigned different charges, representing more closely the actual [[Amortized Analysis|amortized cost]] of each operation.

---
## See Also
- [[Amortized Analysis]]
- [[Aggregate Method (Amortized Analysis)]]
- [[Accounting Method (Amortized Analysis)]]
- [[Potential Method (Amortized Analysis)]]
- [[Dynamic Array]]

