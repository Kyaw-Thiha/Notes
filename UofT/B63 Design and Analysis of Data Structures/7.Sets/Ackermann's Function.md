# Ackermann's Function
Let $k \geq 1, \ j \geq 1$.
We define [[Ackermann's Function]] to be
$$
\begin{cases}
A_{0}(j) = j + 1 \\[8pt]
A_{k}(j) = \underbrace{A_{k-1}(A_{k-1}( \dots (j) \dots ))}_{\text{We do this } j+1 \text{ times}}
\end{cases}
$$
$A_{k}(j)$ increases when $j$ or $k$ increases.

---
## Example
Let $j=1$.
Then,
$$
\begin{align}
A_{0}(1) &= 1 + 1 \\  
&= 2 \\[10pt]

A_{1}(1) &= A_{0}(A_{0}(1)) \\  
&= A_{0}(2) \\
&=2 + 1 \\
&=3 \\[10pt]

A_{2}(1) &= A_{1}(A_{1}(1))  \\
&= A_{1}(3) \\
&= A_{0}(A_{0}(A_{0}(A_{0}(3)))) \\
&= A_{0}(A_{0}(A_{0}(4))) \\
&= A_{0}(A_{0}(5)) \\
&= A_{0}(6) \\
&= 7 \\[10pt]

A_{3}(1) &= A_{2}(A_{2}(1))  \\
&= A_{2}(7) \\
&= \underbrace{A_{1}(\dots A_{1}}_{8 \text{ times}}(7) \dots) \\
&= 2047 \\[10pt]

A_{4}(1) &> 10^{80}
\end{align}
$$

Note that the [[Ackermann's Function|Ackermann's function]] grows very quickly.

---
## Inverse Ackermann's Function
We can define the [[#Inverse Ackermann's Function|inverse Ackermann's Function]], denoted by $\alpha(n)$, as
$$
\alpha(n) = \min\{ k: A_{k}(1) \geq n \}
$$
$\text{E.g:}$ $\alpha(10^{80}) = 4$.

> **Note**: The [[#Inverse Ackermann's Function|inverse Ackermann's Function]] grows very slowly.
> $\alpha(n)$ is practically a constant.

---
## See Also
- [[Disjoint Set]]
- [[Time Complexity]]
- [[Amortized Analysis]]
- [[Kruskal's Algorithm]]
