# LQR Controller
#robotics/controller/lqr  
`Linear Quadratic Regulator (LQR)` controller that minimizes quadratic cost function to balances state error and control effort.

![LQR Controller|500](https://www.mathworks.com/discovery/optimal-control/_jcr_content/mainParsys/columns_1092544956/d9ef23e9-01a7-49fe-9313-4c76fcdedf14/image_copy_copy.adapt.full.medium.jpg/1765106504795.jpg)

---
`Linear State-Space System`
Recall from [[Controller]] that the linear state system is represented as
$$
\dot{x} = Ax + Bu
$$
where
- $\vec{x}$ is the `state vector` $(\text{position, velocity, angle, etc})$
- $\vec{u}$ is the `control input` $(\text{force, torque, voltage})$
- $A$ and $B$ are `system matrices`

---
`Cost Function`
`LQR` minimizes the [[Loss Function|quadratic cost function]]:
$$
J = \int^{\infty}_{0} (x^T Q x + u^T R u) \ dt
$$
where
- $\vec{x}$ is the `state vector`
- $\vec{u}$ is the `control input`
- $Q$ is the `error penalizing matrix`
- $R$ is the `aggressive control penalizing matrix`

Large $Q$ prioritizes good accuracy.
Large $R$ prioritizes good control effort.

---
### Hamilton–Jacobi–Bellman(HJB) Principle

`HJB Principle`
> An optimal trajectory remains optimal after a small amount of time.

This means instead of minimizing the whole integral at once, we can minimize it recursively.


`Value Function`
The value function can be defined as
$$
V(x) = \min_{u} 
\int^{\infty}_{0} (x^T Q x + u^T R u) \ dt
$$

Using the following properties:
1. [[First-Order Taylor Expansion]]
2. At time $t=0$, $x(0) = x$
   At time $t=dt$, $x(dt) = x + \dot{x} \ dt$

We can handle the integral as
$$
\begin{align}
&\int^{\infty}_{0} (x^T Q x + u^T R u) \ dt \\[6pt]
&= \int^{\infty}_{dt} (x^T Q x + u^T R u) \ dt
+ \int^{dt}_{0} (x^T Q x + u^T R u) \ dt \\[6pt]
&= \int^{dt}_{0} (x^T Q x + u^T R u) \ dt
+ \int^{\infty}_{dt} (x^T Q x + u^T R u) \ dt \\[6pt]
&= (x^T Q x + u^T R u) \ dt
+ \int^{\infty}_{dt} (x^T Q x + u^T R u) \ dt  
\\[6pt]

&= (x^T Q x + u^T R u) \ dt
+ V(x(dt)) \ dt \\[6pt]

&= (x^T Q x + u^T R u) \ dt
+ V(x + \dot{x}) \ dt
\end{align}
$$

---
`Minimizing w.r.t u`
Deriving it,
$$
\begin{align}
V(x) 
&= \min_{u} 
\int^{\infty}_{0} (x^T Q x + u^T R u) \ dt \\[6pt]


&= \min_{u} [ \ (x^T Q x + u^T R u) \ dt  
+ V(x + \dot{x} \ dt) \ ] \\[6pt]

&\approx \min_{u} [ \ (x^T Q x + u^T R u) \ dt  
+ V(x) + \nabla V(x)^T \dot{x} \ dt \ ] \\[6pt]

0 &= \min_{u} [ \ x^T Q x + u^T R u  
+ \nabla V(x)^T (Ax + Bu) \ dt \ ] \\[6pt]
\end{align}
$$

`HJB Expression`
Since the system is quadratic, we get
$$
V(x) = x^T P x \ \quad 
, \text{ where } P = P^T
$$
Then, the gradient is
$$
\nabla V(x) = 2 \ P x
$$
Substituting it in,
$$
\begin{align}
0 &= \min_{u} [ \ x^T Q x + u^T R u  
+ \nabla V(x)^T (Ax + Bu) \ dt \ ] \\[6pt]

0 &= \min_{u} [ \ x^T Q x + u^T R u  
+ (2Px)^T (Ax + Bu) \ ] \\[6pt]

0 &= \min_{u} [ \ x^T Q x + u^T R u  
+ 2x^T P \ Ax + 2x^T P \ Bu \ ] \\[6pt]
\end{align}
$$

`Optimization`
Minimizing $w.r.t$ $u$,
$$
\begin{align}
&\min_{u} [ \ x^T Q x + u^T R u  
+ 2x^T P \ Ax + 2x^T P \ Bu \ ] \\[6pt]

&\implies \min_{u} [ \ u^T R u  
+ 2x^T P \ Bu \ ] \\[6pt]
 
&\implies \frac{\partial}{\partial u}  
( \ u^T R u + 2x^TP \ Bu \ ) = 0 \\[6pt]

&\implies 2Ru + 2B^TPx = 0 \\[6pt]

&\implies u = -R^{-1} B^T Px
\end{align}
$$

So, the optimal control can be achieved by
$$
\boxed{ \ u = -Kx, \quad \text{where } K = R^{-1}B^TP \ }
$$

---
### Algebraic Riccati Equation (ARE)
Substituting $u = -R^{-1} B^T Px$ back into the HJB expression,

- The `control cost term` becomes
$$
u^TRu 
= (x^TPBR^{-1}) \ R \ (R^{-1}B^TPx)
= (x^TPBR^{-1}) \ (B^TPx)
$$
- The `cross term` becomes
$$
2x^TPBu
= 2x^TPB (-R^{-1} B^T Px)
= -2x^TPB R^{-1}B^TP x
$$

Hence,
$$
\begin{align}
0 &= x^T Q x + u^T R u  
+ 2x^T P \ Ax + 2x^T P \ Bu \\[6pt]

0 &= x^T Q x + (x^TPBR^{-1}) \ (B^TPx)  
+ 2x^T P \ Ax + -2x^TPB R^{-1}B^TP x \\[6pt]

0 &= x^T Q x + 2x^T P \ Ax  
+ x^TPBR^{-1}B^TPx  
-2x^TPB R^{-1}B^TP x \\[6pt]

0 &= x^T Q x + 2x^T P \ Ax  
-x^TPB R^{-1}B^TP x \\[6pt]

0 &= x^T Q x + x^T \ (PA + A^TP) \ x  
-x^TPB R^{-1}B^TP x \\[6pt]

0 &= x^T \ (Q + PA + A^TP - PBR^{-1}B^TP) \ x  \\[6pt]

\end{align}
$$

Since this must hold for all $x$, the matrix must be zero
$$
A^TP + PA  - PBR^{-1}B^TP + Q = 0
$$
This is the `Algebraic Riccati Equation (ARE)`

---
## See Also
- [[Controller]]
- [[Bang-Bang Controller]]
- [[PID Controller]]
- [[MPC Controller]]
