# Dynamic Arrays
Each time the array is full, double the size and copy the elements before appending the new element.

---
## Motivation
Consider an array of fixed size and two operations:
- $\text{append()}$: Store an element in the first free position of the array.
- $\text{delete}()$: Remove the element in the last occupied position of the array.

We can use a stack to implement the array.
- The advantage of using a stack is that accessing elements is very efficient.
- The disadvantage is that the size is fixed.

To get around the disadvantage, we can use [[Dynamic Array]].

---
## Algorithm
When trying to append an element to an array that's full, we
- Create a new array that is twice the size of the old one.
- Copy all the elements from the old array into the new one.
- Carry out the append operation.

---
### Amortized Analysis with Aggregate Method
To calculate the [[Amortized Analysis|amortized cost]], think about the cost of performing $n$ operations for appending elements, starting from an empty array of size $1$.

![image|400](https://notes-media.kthiha.com/Dynamic-Arrays/84fbad9fa907d2568e136ab2a0e673df.png)

The amortized cost using [[Aggregate Method (Amortized Analysis)|aggregate method]] is $\frac{2^{n+1} - 1}{2^{n}}$.
This equals to $2 - \frac{1}{2^{n}}$.

---
### Amortized Analysis with Accounting Method
Now lets calculate the [[Amortized Analysis|amortized cost]] with the [[Accounting Method (Amortized Analysis)|accounting method]].
- The cost of appending an element if we don't need to increase the array size is $1$.
- The cost of appending if we do need to increase the array size is $1 + \text{current size of the array}$.
- Therefore, we should charge $3$ for each append.
- Every time we don't have to create a new array and copy elements over, we get a credit of $2$.

Note that the number of credits is never negative.
Furthermore, after $n$ appends, the [[Amortized Analysis|amortized cost]] is $O(n)$.

---
## See Also
- [[Amortized Analysis]]
- [[Aggregate Method (Amortized Analysis)]]
- [[Accounting Method (Amortized Analysis)]]
- [[Potential Method (Amortized Analysis)]]
- [[Dynamic Array]]