# Solving Linear System of Equations
Suppose that we want to solve a linear system of equations
$$
Ax \approx b
$$
with $A\in \mathbb{R}^{N \times d}$, $x \in \mathbb{R}^{d}$ and $b \in \mathbb{R}^{N}$ where $N \geq d$.

When $N > d$, this is an overdetermined system so, the equality may not be satisfied.

---
## Optimization Approach
Formulating it as an optimization problem:
$$
x^{*}
\leftarrow \arg\min_{x \in \mathbb{R}^{d}}
||Ax - b||^{2}_{2}
= (Ax - b)^{T} (Ax-b)
$$
We can then use numerical optimizers like [[Gradient Descent|gradient descent]].
As the gradient of $(Ax - b)^{T}(Ax - b)$ is
$$
A^{T}_{d \times N}(A_{N \times d} \ x - b)
$$
the [[Gradient Descent|gradient descent]] procedure would be
$$
x_{k+1} \leftarrow x_{k} - \alpha A^{T} (Ax_{k} - b)
$$
We can use more advanced optimization techniques too.
This approach finds a $x^{*}$ that minimizes the squared error loss function.

---
## Direct Approach
Solve for the zero of the gradient
$$
\begin{align}
&A^{T}(A \ x - b) = 0 \\[6pt]
&\implies A^{T}Ax = A^{T}b \\[6pt]
&\implies x^{*} = (A^{T}A)^{-1} A^{T}b
\end{align}
$$
assuming the invertibility of $A^{T}A$.

For this approach, we need to be able to invert the matrix $A^{T}A$.

---
## Fixed-Point Iteration
We can rewrite $Ax=b$
$$
(\mathbf{I} - A)x + b = x
$$
Suppose $N=d$.
This is a form of a [[Fixed Point Iteration (FPI)|fixed-point equation]] 
$$
Lx=x
$$
with $L:\mathbb{R}^{d} \to \mathbb{R}^{d}$ being the mapping
$$
L: x \mapsto (\mathbf{I} - A)x + b
$$
Suppose $L$ is a [[Contraction Mapping|contraction mapping]] which isn't always the case.
By [[Banach Fixed Point Theorem]], the iterative procedure is
$$
x_{k+1} \leftarrow Lx_{k}
= (\mathbf{I} - A) x_{k} + b
$$
converges to $x^{*}$, which is the solution of $Ax^{*} = b$.

It is also possible to define a slightly modified version of
$$
x_{k+1} \leftarrow
(1-\alpha) \ x_{k} + \alpha L \ x_{k}
$$
This is similar to the iterative procedure in [[Stochastic Approximation(SA)]] without noise..

---
