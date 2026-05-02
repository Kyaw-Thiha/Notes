# Running Time of Algorthms
Suppose we were to count each step.
We can do
1. read, write vars: `1 each`
2. method call: `+1 step to evaluate each arg`
   `+ steps to execute the method`
3. return statement: `+1 step`
4. if statement, while statement (not entire loop):
   `+1 step to eval exit condition`
5. assignment statement: `+1 step to eval both sides`
6. arithmetic comparism, boolean operators: 
   `+1 step to eval each operand`
7. array access: `+1 step to eval index`
8. constants: `Free`

---
### Example
Consider this function for insertion sort.
**Pre-Condition**: $A$ is an array of ints
**Post-Condition**: $A$ is sorted in non-decreasing order

![image|400](https://notes-media.kthiha.com/Running-Time-of-Algorithms/e05058186c5c65d9c5f3f68a02ab1904.png)

---
## Finding Time Complexity
To find its time complexity, suppose $A$ has $n$ elements.

- $\text{Line-1}$ takes $2$ steps.
- Outer-loop $(\text{lines-}2,3,4,8,9)$ runs $n-1$ times.
  Therefore, outer-loop takes $2 \ O(n-1)$ steps.
- However, $\text{line-2}$ is evaluated one last time, 
  and it takes $5$ steps.
- Each time the inner loop $(\text{lines }5-7)$ runs, $j$ goes from $i$ to $1$.
  Therefore, it takes $19i$ steps.
- However, $\text{line-}5$ is executed once more, which takes $9$ steps.

In total, the inner loop takes
$$
\begin{align}
\sum^{n-1}_{i=1} 19i + 9
&= \sum ^{n-1}_{i=1} 19i + \sum ^{n-1}_{i=1} 9  
\\[6pt]
&= \frac{19n \ (n-1)}{2} + 9(n-1) \\[6pt]
&= \frac{19n^{2}}{2} - \frac{19n}{2} + 9n - 9 \\[6pt]
&= \frac{19n^{2}}{2} - \frac{n}{2} - 9
\end{align}
$$
Hence, the total is 
$$
\begin{align}
\text{Total}  
&= 2 + 2O(n-1) + 5 + \frac{19n^{2}}{2}  
- \frac{n}{2} - 9 \\[6pt]
&= \frac{19n^{2}}{2} + \frac{31n}{2} - 22
\end{align}
$$

---
### Introducing Big O
Note that if we were to ran the code on an older computer, it would take more time.

The **quadratic polynomials** are of order $n^{2}$.
The **cubic polynomials** are of order $n^{3}$.
$4n \ \log(n) + 2n + 10$ is of order $n \ \log(n)$.

**Example**:
Show that $12n^{2} + 10n + 10$ is of order $n^{2}$.
$$
\begin{align}
12n^{2} + 10n + 10  
&\leq 12n^{2} + 10n^{2} + 10 \\[6pt]
&= 22n^{2} + 10
\end{align}
$$
For all $n\geq 10$:
$$
\begin{align}
22n^{2} + 10
&\leq 22n^{2} + n \\[6pt]
&\leq 22n^{2} + n^{2} \\[6pt]
&\leq 23n^{2} \\[6pt]
\end{align}
$$

---
