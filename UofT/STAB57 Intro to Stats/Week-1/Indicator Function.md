# Indicator Function
[[Indicator Function|Indicator function]] of $A$, written $I_{A}$, is defined as 
$$
I_{A}(s)
= \begin{cases}
1, \ \text{if } s \in A \\[6pt]
0, \ \text{if } s \notin A
\end{cases}
$$

---
## Relation to PDF
Note that all PDFs are positive. So, when $x \leq 0$, it could be $0$.

### Exponential Distribution
Look at the following exponential distribution.
$$
f(x) = \begin{cases}
\lambda e^{-\lambda x} & , \ x > 0 \\[6pt]
0 & , \ \text{otherwise}
\end{cases}
$$
Hence, we can redefine it as
$$
f(x) = \lambda e^{-\lambda x} \cdot \mathbf{I}(x > 0)
$$

### Uniform Distribution
Likewise, look at the following uniform distribution.
$$
f(x) = \begin{cases}
\frac{1}{b-a} & , \ a \leq x \leq b \\[6pt]
0 & , \ \text{otherwise}
\end{cases}
$$
Hence, we can redefine it as 
$$
f(x) = \frac{1}{b-1} \cdot \mathbf{I}(x > 0)
$$

---