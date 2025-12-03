# Vandermonde Example
#numerical-methods/interpolation/vendermonde 

`Question`
Compute the `quadratic polynomial` interpolating $\{ (0,3), \ (1,7), \ (2, 37) \}$ using the [[Vandermonde Theorem|Method of Undetermined Coefficients]].

---

`Solution`

Recall that the `Monomial Basis` is
$$
p(x) = \sum^{n}_{i=0} a_{i} \ x^{i}
$$
where $n=2$.

To find the $a_{i}$, we'll use the `Vandermonde Matrix`.
$$
\begin{align}
&\begin{bmatrix}
(x_{0})^{0} & (x_{0})^{1} & (x_{0})^{2}  \\
(x_{1})^{0} & (x_{1})^{1} & (x_{1})^{2}  \\
(x_{2})^{0} & (x_{2})^{1} & (x_{2})^{2}  \\
\end{bmatrix}
\begin{bmatrix}
a_{0} \\ a_{1} \\ a_{2}
\end{bmatrix}
=
\begin{bmatrix}
y_{0} \\ y_{1} \\ y_{2}
\end{bmatrix} \\[6pt]
 
&\underbrace{
\begin{bmatrix}
1 & 0 & 0  \\
1 & 1 & 1 \\
1 & 2 & 4
\end{bmatrix}}_{\text{Vandermonde Matrix}}
\begin{bmatrix}
a_{0} \\ a_{1} \\ a_{2}
\end{bmatrix}
= \begin{bmatrix}
3 \\ 7 \\ 37
\end{bmatrix}
\end{align}
$$


---
`Solving Linear Equation`

Let $V$ be the [[Vandermonde Theorem|Vandermonde Matrix]].
We can use $PV = LU$ ([[LU Factorization]]) to solve for $\bar{a}$.

Forming the lower triangular matrix $L_{1}$.
$$
L_{1} = \begin{bmatrix}
1 & 0 & 0  \\
-1 & 1 & 0  \\
-1 & 0 & 1
\end{bmatrix}
$$

$$
\begin{align}
L_{1}V &= \begin{bmatrix}
1 & 0 & 0 \\
-1 & 1 & 0 \\
-1 & 0 & 0
\end{bmatrix}
\begin{bmatrix}
1 & 0 & 0  \\
1 & 1 & 1 \\
1 & 2 & 4
\end{bmatrix} \\[6pt]
&= \begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 1 \\
0 & 2 & 4
\end{bmatrix}
\end{align}
$$

Permuting the matrix for lower triangular matrix $L_{2}$.
$$
P_{1} = \begin{bmatrix}
1 & 0 & 0 \\
0 & 0 & 1 \\
0 & 1 & 0
\end{bmatrix}
$$
$$
\begin{align}
P_{1}L_{1}V
&= \begin{bmatrix}
1 & 0 & 0 \\
0 & 0 & 1 \\
0 & 1 & 0
\end{bmatrix}
\begin{bmatrix}
1 & 0 & 0  \\
0 & 1 & 1 \\
0 & 2 & 4
\end{bmatrix} \\[6pt]
&= 
\begin{bmatrix}
1 & 0 & 0 \\
0 & 2 & 4 \\
0 & 1 & 1
\end{bmatrix}
\end{align}
$$

Forming the lower triangular $L_{2}$.
$$
L_{2} = \begin{bmatrix}
1 & 0 & 0 \\
0 & 2 & 4 \\
0 & -\frac{1}{2} & 1
\end{bmatrix}
$$
$$
\begin{align}
L_{2}P_{1}L_{1}V
&= \begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & -\frac{1}{2} & 1
\end{bmatrix}
\begin{bmatrix}
1 & 0 & 0 \\
0 & 2 & 4 \\
0 & 1 & 1
\end{bmatrix} \\[6pt]
&=  
\underbrace{\begin{bmatrix}
1 & 0 & 0 \\
0 & 2 & 4 \\
0 & 0 & -1
\end{bmatrix}}_{u}
\end{align}
$$

`LU Form`

Reording the terms, u$L_{2}P_{2}L_{1}\ V$ is equivalent to $L_{2} \ \underbrace{P_{2}L_{1}P_{1}}_{\hat{L}_{1}} P_{1} \ V$
where 
$$
\hat{L}_{1} 
= P_{1}L_{1}P_{1}
= \begin{bmatrix}
1 & 0 & 0  \\
-1 & 1 & 0  \\
-1 & 0 & 1
\end{bmatrix} 
$$

$$
\begin{align}
L  
&= \hat{L}_{1}^{-1} \cdot L_{2}^{-1} \\[6pt]
&= \hat{L}_{1}^{-1} + \hat{L}_{2}^{-1} - I \\[6pt]
&= \begin{bmatrix}
1 & 0 & 0 \\
1 & 1 & 0 \\
1 & \frac{1}{2} & 1
\end{bmatrix}
\end{align}
$$
$$
P = P_{1}
= \begin{bmatrix}
1 & 0 & 0  \\
0 & 0 & 1 \\
0 & 1 & 0
\end{bmatrix}
$$
Now that we have $P, L$ and $U$, we can start solving
$$
\begin{align}
&V\vec{a} = \vec{b} \\[6pt]
&\leftrightarrow PV \ \vec{a} = P \vec{b} \\[6pt]
&\leftrightarrow LU \ \vec{a} = P \vec{b} \\[6pt]
&\leftrightarrow L\vec{d} = P \vec{b}
\end{align}
$$

`Forward Elimination`

In $L\vec{d} = P\vec{b}$, we solve for $\vec{d}$.
In $U\vec{a} = \vec{d}$, we solve for $\vec{a}$.

$$
\begin{align}
P\vec{b}
&= \begin{bmatrix}
1 & 0 & 0 \\
0 & 0 & 1 \\
0 & 1 & 0
\end{bmatrix}
\begin{bmatrix}
3 \\ 7  \\ 37
\end{bmatrix} \\[6pt]
&= \begin{bmatrix}
3 \\ 37 \\ 7
\end{bmatrix}
\end{align}
$$

Using it, we can solve
$$
\begin{align}
L\vec{d} &= P\vec{b} \\[6pt]

\begin{bmatrix}
1 & 0 & 0 \\
1 & 1 & 0 \\
1 & \frac{1}{2} & 1
\end{bmatrix}
\begin{bmatrix}
d_{1} \\ d_{2} \\ d_{3}
\end{bmatrix}
&= \begin{bmatrix}
3 \\ 37 \\ 7
\end{bmatrix} \\[6pt]

d_{1} &= 3 \\
d_{1} + d_{2} &= 37  \implies d_{2} = 34 \\
d_{1} + \frac{d_{2}}{2} + d_{3} &= 7 \implies d_{3} = -13 \\[6pt]

&\implies \vec{d}
= \begin{bmatrix}
3 \\ 34 \\ -13
\end{bmatrix}
\end{align}
$$

`Backward Substitution`
$$
\begin{align}
&U\vec{a} = \vec{d} \\[6pt]

&\begin{bmatrix}
1 & 0 & 0 \\
0 & 2 & 4 \\
0 & 0 & -1
\end{bmatrix}
\begin{bmatrix}
a_{0} \\ a_{1} \\ a_{2}
\end{bmatrix}
= \begin{bmatrix}
3 \\  34 \\ -34
\end{bmatrix} \\[6pt]

&\implies \\
&a_{0} = 3 \\
&a_{2} = 13 \\[6pt]

&\implies \\
&2a_{1} + 4a_{2} = 34 \\
&2a_{1} + 52 = 34 \\
&2a_{1} = -18 \\
&a_{1} = -9
\end{align}
$$

$$
\vec{a} = \begin{bmatrix}
3 \\ -9 \\ 13
\end{bmatrix}
$$

`Final Answer`
$$
\begin{align}
P(x)
&= \sum^{2}_{i=0} a_{i} \ x^i \\
&= 3 -9x + 13x^2
\end{align}
$$

---

`Checking`
We can check by 
$$
\begin{align}
P(0)
&= 3 - 9(0) + 13(0)^2 \\[6pt]
&= 3 \\[6pt]
&= y(0)
\end{align}
$$
$$
\begin{align}
P(1)
&= 3 - 9(1) + 13(1)^2 \\[6pt]
&= 3 \\[6pt]
&= y(0)
\end{align}
$$
$$
\begin{align}
P(2)
&= 3 - 9(2) + 13(2)^2 \\[6pt]
&= 37 \\[6pt]
&= y(2)
\end{align}
$$

---
## See Also
- [[Vandermonde Theorem]]