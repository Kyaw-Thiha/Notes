# Newton Basis
#numerical-methods/interpolation/newton 

`Newton Basis` is also called `Divided Differences`.

For a simple interpolation $P(x_{i}) = y_{i}$, $i=0, 1, \dots, n$, we look for an interpolating of the form
$$
p(x) = a_{0} + a_{1}(x-x_{0}) + a_{2}(x-x_{0})(x-x_{1}) + \dots + a_{n}(x-x_{0})(x-x_{1}) \dots (x-x_{n-1})
$$

Converting into a matrix, we get
$$
\begin{bmatrix}
1 & 0 & 0 & \dots & 0 \\[6pt]
1 & x_{1}-x_{0} & 0 & \dots & 0 \\[6pt]
1 & x_{2}-x_{0} & (x_{1} - x_{0})(x_{2}-x_{1})  & \dots & 0 \\[6pt]
\vdots & \vdots & \vdots & & \vdots \\[6pt]
1 & x_{n}-x_{0} & (x_{n}-x_{0})(x_{n}-x_{1})  
& \dots & \prod^{n-1}_{i=0} (x_{n} - x_{i})
\end{bmatrix}
\begin{bmatrix}
a_{0} \\[6pt]
a_{1} \\[6pt]
\vdots \\[6pt]
a_{n} \\[6pt]
\end{bmatrix}
=
\begin{bmatrix}
y_{0} \\[6pt]
y_{1} \\[6pt]
\vdots \\[6pt]
y_{n} \\[6pt]
\end{bmatrix}
$$

This is a `lower triangular matrix`, meaning that no factorization is involved

$$
\begin{align}
a_{0} = y_{0} \\[6pt]
a_{1} = \frac{y_{1} - y_{0}}{x_{1} - x_{0}} \\[6pt]
a_{2} = \frac{\frac{y_{2}-y_{1}}{x_{2}-x_{1}} - 
 \frac{y_{1} - y_{0}}{x_{1}-x_{0}}} 
{x_{2} - x_{0}}
\end{align}
$$

`Divided Differences`: $Y[x_{i}] = y(x_{i}) = y_{i}$

$$
Y[x_{i+k}, \ \dots, \  x_{i}]
= \frac{Y[x_{i+k}, \ \dots, \ x_{i+1}] - Y[x_{i+k-1}, \ \dots, \ x_{i}]}
{x_{i+k} - x_{i}}
$$

`E.g.`: $Y[x_{2}, \ x_{1}, \ x_{0}] = \frac{Y[x_{2}, \ x_{1}] - Y[x_{1}, \ x_{0}]}{x_{2} - x_{0}}$

---

`Newton Polynomial`
$$
p(x) = y[x_{0}]
+ (x-x_{0}) \ Y[x,x_{0}] + \dots 
+ (x-x_{0})(x-x_{1}) \dots 
(x-x_{n-1}) \ Y[x_{n}, \dots, x_{0}]
$$

Then, $p(x) \in P_{n}$ and $p(x_{i}) = y_{i}$, $i=0, 1, 2, \dots, n$

---
## Example
`Question`: Find a $P \in P_{3}$ $s.t.$ $P(0)=1$, $P(1)=3$, $P(2)=9$, $P(3)=25$

`Soln`

![[Newton Polynomial.png]]

$$
\begin{align}
P(x)
&= Y[x_{0}]  \\
&+ (x - x_{0})Y[x_{1}, x_{0}] \\
&+ (x-x_{0})(x-x_{1}) \ Y[x_{2},\ x_{1},\ x_{0}] \\
&+ (x-x_{0})(x-x_{1})(x-x_{2}) \ Y[x_{3}, \dots, x_{0}] \\[6pt]
&= 1 + 2x + 2x(x-1) + x(x-1)(x-2)
\end{align}
$$
Read coefficients from top of the triangle

---
`Relation between divided differences and derivatives`

Consider $Y[x_{1}, x_{0}] = \frac{Y(x_{1}) - Y(x_{0})}{x_{1} - x_{0}}$
$$
\begin{align}
\lim_{ x_{1} \to x_{0} } Y[x_{1}, \ x_{0}]
&= \lim_{ x_{1} \to x_{0} } \frac{Y(x_{1}) - Y(x_{0})}{x_{1} - x_{0}} \\[6pt]
&= Y'(0)\quad ,\text{ provided that } Y'(x_{0}) \text{ exists}
\end{align}
$$

Consider $Y[x_{2}, \ x_{1},\ x_{0}] = \frac{Y[x_{2}, x_{1}] - Y[x_{1}, x_{0}]}{x_{2} - x_{1}}$.

$$
\lim_{ \substack{x_{2} \to x_{0} 
\\ x_{1} \to x_{0}}  } 
Y[x_{2}, x_{1}, x_{0}]
= \frac{Y''(x_{0})}{2!}
$$

In general, we can show that 
$$
\lim_{ \substack{x_{k} \to x_{0} 
\\ x_{k-1} \to x_{0} \\ \vdots \\  x_{1} \to x_{0}} } 
Y[x_{k}, \dots, x_{0}]
= \frac{y^{(k)}(x_{0})}{k!}
$$

---
`Example`

`Question`: Find $P \in P_{4}$ $s.t.$ $P(0) = 0$, $P(1) = 1$, $P'(1) = 1$, $P''(1) = 2$ and $P(2)=6$

`Solution`
![[Newton Polynomial-2.png]]

$$
\begin{align}
P(x)
&= Y[0] + x \ Y[0, 1] + x(x-1) \ Y[1,1,0]\\
&+ x(x-1)^2 \ Y[1, 1, 1, 0]  \\
&+ x(x-1)^3 \ Y[2, 1, 1, 1, 0] \\[6pt]
&= \underbrace{0 + x + x(x-1)^2 + x(x-1)^3}_{\text{Read the coefficients from top of triangle}}
\end{align}
$$

---
## See Also
- [[Polynomial Interpolation]]
- [[Lagrange Matrix]]
