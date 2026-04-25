# Lagrange Matrix
#numerical-methods/interpolation/lagrange 

Write the [[Polynomial Interpolation|Interpolating Polynomial]] as 
$$
P(x) = \sum^{n}_{i=0} y_{i} \ L_{i}(x) 
$$
where $L_{i}(x)$ is the `Lagrange Basis Polynomial`
$$
L_{i}(x) 
= \prod^{n}_{j=i, \ j\neq i}
\frac{x - x_{j}}{x_{i} - x_{j}}
$$

For points $(x^{(0)}, x^{(1)}, \dots, x^{(m)})$, we can write it in matrix as
$$
\begin{bmatrix}
P(x^{(0)}) \\[6pt]
P(x^{(1)}) \\[6pt]
\vdots \\[6pt]
P(x^{(m)})
\end{bmatrix}
= 
\underbrace{
\begin{bmatrix}
L_{0}(x^{(0)}) & L_{1}(x^{(0)}) & \dots & L_{n}(x^{(0)})  
\\[6pt]
L_{0}(x^{(1)}) & L_{1}(x^{(1)}) & \dots & L_{n}(x^{(1)})  
\\[6pt]
\vdots & \vdots & & \vdots \\[6pt]
L_{0}(x^{(m)}) & L_{1}(x^{(m)}) & \dots & L_{n}(x^{(m)})  
\\[6pt]
\end{bmatrix}}_{\text{Lagrange Matrix}}
\begin{bmatrix}
y_{0}  \\[6pt]
y_{1}  \\[6pt]
\vdots \\[6pt]
y_{n}  \\[6pt]
\end{bmatrix}
$$
Each element $[L]_{ij} = L_{j}(x^{(i)})$ 

---
`Idea`
For a simple `interpolation problem` $p(x_{i}) = y_{i}, \ i=0,1,\dots,n$, consider the following basis:
$$
\begin{align}
L_{i}(x)
&= \prod^n_{j=0, \ j\neq i} \frac{x - x_{j}}{x_{i} - x_{j}}  
\quad \text{for } i=0,1,2,\dots,n \\[6pt]
&= \left( \frac{x - x_{0}}{x_{i} - x_{0}} \right)
\dots
\underbrace{
\left( \frac{x - x_{i-1}}{x_{i} - x_{i-1}} \right)
\left( \frac{x - x_{i+1}}{x_{i} - x_{i+1}} \right)
} 
_{\text{Notice that we skipped } \frac{x - x_{i}}{x_{i} - x_{i}} }
\dots
\left( \frac{x - x_{n}}{x_{i} - x_{n}} \right)
\end{align}
$$

`Proof`
Let $L_{i}(x) \in P_{n}$ 
Consider $l_{i}(x_{j})$.

If $j=i$, $L_{i}(x_{j}) = 1$
$$
\begin{align}
L_{i}(x_{j})  
&= \prod^{n}_{j=i, \ j\neq i} \frac{x_{i} - x_{j}}{x_{i} - x_{j}} \\[6pt]
&= \left( \frac{x_{i} - x_{0}}{x_{i} - x_{0}} \right)
\dots  
\left( \frac{x_{i} - x_{i-1}}{x_{i} - x_{i-1}} \right)
\left( \frac{x_{i} - x_{i+1}}{x_{i} - x_{i+1}} \right)
\dots  
\left( \frac{x_{i} - x_{n}}{x_{i} - x_{n}} \right)  
\\[6pt]
&= 1
\end{align}
$$

If $j \neq i$, $l_{i}(x_{j}) = 0$
$$
\begin{align}
L_{i}(x_{j})  
&= \prod^{n}_{j \neq i, \ j\neq i} \frac{x_{j} - x_{j}}{x_{i} - x_{j}} \\[6pt]
&= \left( \frac{x_{j} - x_{0}}{x_{i} - x_{0}} \right)
\dots  
\left( \frac{x_{j} - x_{i-1}}{x_{i} - x_{i-1}} \right)
\left( \frac{x_{j} - x_{i+1}}{x_{i} - x_{i+1}} \right)
\dots  
\left( \frac{x_{j} - x_{n}}{x_{i} - x_{n}} \right)  
\\[6pt]
&= 0
\end{align}
$$
One of the products will be $0$ as $0 \leq j \leq n$ and $j \neq n$.
Hence, the entire product will be $0$.

To summarize,
$$
L_{i}(x_{j}) = \begin{cases}
1, & \text{if } i=j \\
0, & \text{if } i \neq j
\end{cases}
$$

---
`Properties`

The `Lagrange polynomial` is free to construct, but very expensive to evaluate at non-interpolation points.

With the `basis function`, we can write out the interpolating polynomial for free.

$$
p(x) = \sum^{n}_{i=0} l_{i}(x) \ y_{i}
$$
where $l_{i}(x) = 1$, as stated previously.

---
## See Also
- [[Vandermonde Theorem]]
- [[Polynomial Interpolation]]
