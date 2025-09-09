#math #numerical-methods/linear-systems

## Linear Systems
Consider $Ax = b$,
- where $A$ is $m$ x $n$ matrix
- $b$ is m-vector
- $x$ is n-vector

## Singular vs Non-Singular
Matrix A is singular if
- A has no inverse
- det(A) = 0
- rank(A) < n
- $Az = 0$ for some vector $z \neq 0$ 

Non-singular: 1 solution
Singular: 0 solution or infinite solutions

For 2D, we can think of it as lines intersecting one another for non-singular, and lines being parallel or exactly the same for singular.

## Solving with Direct Method
This is also known as Gaussian elimination method.

Suppose we are solving for $A.x = b$.

1. First, we should first try to denote $A = L.U$
where
- $L$ is the strictly lower triangle
- $U$ is the upper triangle

2. Secondly substituting $A=L.U$, we get
- $L.U.x = b$

3. Thirdly, since they are now triangle matrix, we can easily solve them in two steps:
- Forward Elimination: $L.d = b$
$$
\begin{bmatrix}
1 & 0 & 0 \\
l_{1} & 1 & 0 \\
l_{2} & l_{3} & 1 \\
\end{bmatrix}
.
\begin{bmatrix}
d_{1} \\
d_{2} \\
d_{3} \\
\end{bmatrix}
=
\begin{bmatrix}
b_{1} \\
b_{2} \\
b_{3} \\
\end{bmatrix}
$$
- Backward Substitution: $U.x = d$
$$
\begin{bmatrix}
u_{1} & u_{2} & u_{3} \\
0 & u_{4} & u_{5} \\
0 & 0 & u_{6} \\
\end{bmatrix}
.
\begin{bmatrix}
x_{1} \\
x_{2} \\
x_{3} \\
\end{bmatrix}
=
\begin{bmatrix}
d_{1} \\
d_{2} \\
d_{3} \\
\end{bmatrix}
$$

## Notes on Direct Method
- Any matrix $A$ can be denoted as $A.x = b$
- Even when matrix $A$ is singular (uninvertible), we can still carry out forward elimination.
  Since matrix $L$ is strictly lower triangular, it is always invertible.
- When matrix $A$ is singular, we cannot do backward substitution. 
  This is because matrix $U$ is uninvertible if and only if matrix $A$ is uninvertible.

## Solving with Iterative Method
Suppose we are solving for $A.x = b$.

1. Firstly, denote $A = L + D + U$,
where
- $L$ is strictly lower triangular
- $D$ is a diagonal matrix
- $U$ is strictly upper triangular

2. Secondly,
$$
A.x = b
$$
$$
\begin{aligned}
&\Leftrightarrow (L + D + U)\,x = b \\
&\Leftrightarrow Lx + Dx + Ux = b \\
&\Leftrightarrow Dx = -(L + U)\,x + b \\
&\Leftrightarrow Dx = c,\ \text{where } c = -(L + U)\,x + b
\end{aligned}
$$

Note that $x$ exists in both left & right side of the equation.
So, we have to solve using iterative method.

3. Thirdly, solve by using iterative method.
$$
D.X^{k+1} = -(L+U).X^k + b
$$
where 
- $k= 0, 1, \dots$ 
- $X^0=0$

We repeat this till $X^{k+1}-X^k < \text{threshold}$ 

## Direct vs Iterative
Direct takes $O(n^3)$ while direct takes $O(n)$
