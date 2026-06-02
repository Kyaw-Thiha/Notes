# Maths behind MPC
#robotics/controller/mpc 

`Given`
Recall that `MPC Controller` uses a linear state system 
$$
x_{k+1} = Ax_{k} + Bu_{k}
$$
and has a quadratic cost function
$$
J = \sum^{N-1}_{k=0} 
(x_{k}^T Q x_{k} + u_{k}^T R u_{k})
+ x_{N}^T \ Q_{f} \ x_{N}
$$

`WTS`
Deriving the optimization problem as
$$
\boxed{ \ \min_{u} \frac{1}{2} u^T H u + f^T u \ 
\quad s.t. \ Gu \leq h}
$$
where 
- $H = 2 \ (B^TQB + R)$ 
- $f = 2B^T \ QA \ x_{0}$

---
## Linear System
The `linear system` at step $k$ can be defined as
$$
x_{k+1} = Ax_{k} + Bu_{k}
$$
Since MPC predicts over a horizon, let $k=0,1,2, \dots, N-1, N$.
Then, the control sequence can be represented as
$$
\vec{u} = \begin{bmatrix}
u_{0} \\ u_{1} \\ \vdots \\ u_{N-1}
\end{bmatrix}
$$
Hence, the state at each step-$k$ can be denoted as
$$
\begin{align}
x_{1} &= Ax_{0} + Bu_{0} \\[6pt]
x_{2} &= A^2x_{0} + ABu_{0} + Bu_{1} \\[6pt]
x_{3} &= A^3x_{0} + A^2Bu_{0} + ABu_{1} + Bu_{2}  
\\[6pt]
\vdots \\[6pt]
x_{N} &= A^Nx_{0} + A^{N-1}Bu_{0} + A^{N-2}Bu_{1}
+ \dots + B \ u_{N-1}
\end{align}
$$
In a general form at step-$k$, 
$$
x_{k} = A^k x_{0} + \sum^{k-1}_{i=0} A^{k-1-i} \ 
B u_{i}
$$

`Vectorizing`
We can vectorize the linear system as
$$
\vec{x} = \mathcal{A} x_{0} + \mathcal{B}\vec{u}
$$
where states and control inputs are
$$
\text{states } \ \vec{x} = \begin{bmatrix}
x_{1} \\ x_{2} \\ \vdots \\ x_{N}
\end{bmatrix}
,\  \text{control inputs } \ \vec{u} = \begin{bmatrix}
u_{0} \\ u_{1} \\ \vdots \\ u_{N-1}
\end{bmatrix}
$$
and the system matrices are
$$
\mathcal{A} = \begin{bmatrix}
A \\ A^2 \\ \vdots \\ A^N
\end{bmatrix}
, \ \mathcal{B} = \begin{bmatrix}
B & 0 & \dots & 0 \\
AB & B & \dots & 0 \\
\vdots & \vdots & \ddots & \vdots  \\
A^{N-1}B & A^{N-2}B & \dots & B
\end{bmatrix}
$$

---
## Cost Function
`Vectorizing the Cost Function`
Recall that the cost function is
$$
J = \sum^{N-1}_{k=0} 
(x^T_{k} Q x_{k} + u_{k}^T R u_{k})
+ x^T_{N} \ Q_{f} \ x_{N}
$$
We can vectorize it as
$$
J = \vec{x}^T \mathbf{Q} \ \vec{x} 
+ \vec{u}^T \mathbf{R} \ \vec{u}
$$
where $\mathbf{Q}$ and $\mathbf{R}$ are block-diagonal matrices
$$
\mathbf{Q} = \begin{bmatrix}
Q & 0 & \dots & 0 \\
0 & Q & \dots & 0  \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \dots  & Q_{f}
\end{bmatrix}
, \ \mathbf{R} = \begin{bmatrix}
R & 0 & \dots & 0 \\
0 & R & \dots & 0 \\
\vdots & \vdots & \ddots & \vdots  \\
0 & 0 & \dots & R
\end{bmatrix}
$$
Substituting in the $\vec{x} = \mathcal{A} x_{0} + \mathcal{B}\vec{u}$, we get
$$
\begin{align}
J &= \vec{x}^T \mathbf{Q} \ \vec{x} 
+ \vec{u}^T \mathbf{R} \ \vec{u} \\[6pt]

&= (\mathcal{A} x_{0} + \mathcal{B}\vec{u})^T 
\mathbf{Q} \  
(\mathcal{A} x_{0} + \mathcal{B}\vec{u})
+ \vec{u}^T \mathbf{R} \ \vec{u} \\[6pt]

&= \mathcal{A}^T x_{0}^T \ \mathbf{Q}
\ \mathcal{A} x_{0}
+ 2 \ \mathcal{A}^T x_{0}^T \mathcal{B}^T\vec{u}^T
\ \mathbf{Q}
\ \mathcal{A} x_{0} \mathcal{B}\vec{u}
+ \mathcal{B}^T\vec{u}^T \ \mathbf{Q}  
\ \mathcal{B}\vec{u}
+ \vec{u}^T \mathbf{R} \ \vec{u} \\[6pt]

&= x_{0}^T \mathcal{A}^T \ \mathbf{Q}
\ \mathcal{A} x_{0}
+ 2 \  x_{0}^T \mathcal{A}^T
\ \mathbf{Q}
\ \mathcal{B}\vec{u}
+ \vec{u}^T \mathcal{B}^T \ \mathbf{Q}  
\ \mathcal{B}\vec{u}
+ \vec{u}^T \mathbf{R} \ \vec{u} \\[6pt]

&= \vec{u}^T \mathbf{R} \ \vec{u} 
+ \vec{u}^T \mathcal{B}^T \ \mathbf{Q}  
\ \mathcal{B}\vec{u} 
+ 2 \  x_{0}^T \mathcal{A}^T
\ \mathbf{Q}
\ \mathcal{B}\vec{u} 
+ x_{0}^T \mathcal{A}^T \ \mathbf{Q}
\ \mathcal{A} x_{0}
\\[6pt]

&= \vec{u}^T  
(\mathcal{B}^T  \ \mathbf{Q}  \ \mathcal{B} + \mathbf{R})
\vec{u} 
+ 2 \  x_{0}^T \mathcal{A}^T
\ \mathbf{Q}
\ \mathcal{B}\vec{u} 
+ x_{0}^T \mathcal{A}^T \ \mathbf{Q}
\ \mathcal{A} x_{0}
\\[6pt]
\end{align}
$$

`Optimization`
Recall that our vectorized cost function is
$$
J = \vec{u}^T  
(\mathcal{B}^T  \ \mathbf{Q}  \ \mathcal{B} + \mathbf{R})
\vec{u} 
+ 2 \  x_{0}^T \mathcal{A}^T
\ \mathbf{Q}
\ \mathcal{B}\vec{u} 
+ x_{0}^T \mathcal{A}^T \ \mathbf{Q}
\ \mathcal{A} x_{0}
$$
Since the last term does not depends on $u$, we get
$$
J = \vec{u}^T  
(\mathcal{B}^T  \ \mathbf{Q}  \ \mathcal{B} + \mathbf{R})
\vec{u} 
+ 2 \  x_{0}^T \mathcal{A}^T
\ \mathbf{Q}
\ \mathcal{B}\vec{u} 
$$

Minimizing it, we get
$$
\boxed{ \ \min_{u} \frac{1}{2} u^THu + f^Tu \ }
$$
where 
- $H = 2 \ (B^TQB + R)$ 
- $f = 2B^T \ QA \ x_{0}$

---
`Constraints`
Recall that 
- Control Constraints: $u_{min} \leq u_{k} \leq u_{max}$, $\forall k = 0, 1, \dots, N-1$
- State Constraints:  $x_{min} \leq x_{k} \leq x_{max}$, $\forall k=1, 2, \dots, N$

Note that for sake of simplicity, $u_{k} \in \mathbb{R}$ and $x_{k} \in \mathbb{R}$ will be considered 1 dimensional.

`Input Control Constraints`
First, we redefine the constraints as
$$
\vec{u}_{min} = \begin{bmatrix}
u_{min} \\ u_{min} \\ \vdots \\ u_{min}
\end{bmatrix} 
, \ \vec{u}_{max} = \begin{bmatrix}
u_{max} \\ u_{max} \\ \vdots \\ u_{max}
\end{bmatrix} 
$$
Hence, we now get $\vec{u} \leq \vec{u}_{max}$ and $\vec{u} \leq \vec{u}_{min}$.

Vectorizing the input control constraints, we get
$$
G_{u} \ \vec{u} \leq h_{u}
$$
where
$$
G_{u} = \begin{bmatrix}
I_{N} \\ -I_{N}
\end{bmatrix} \in \mathbb{R}^{2N \times N}
, \quad h_{u} = \begin{bmatrix}
u_{max} \\ u_{min}
\end{bmatrix} \in \mathbb{R}^{2N}
$$

`State Constraints`
Likewise, we redefine the constraints as
$$
\vec{x}_{min} = \begin{bmatrix}
x_{min} \\ x_{min} \\ \vdots \\ x_{min}
\end{bmatrix} 
, \ \vec{x}_{max} = \begin{bmatrix}
x_{max} \\ x_{max} \\ \vdots \\ x_{max}
\end{bmatrix} 
$$
Hence, we now get $\vec{x} \leq \vec{x}_{max}$ and $\vec{x} \leq \vec{x}_{min}$.
Substituting in the $\vec{x} = \mathcal{A} x_{0} + \mathcal{B}\vec{u}$, we get
$$
\mathcal{A} x_{0} + \mathcal{B}\vec{u} \leq x_{max}
\implies 
\mathcal{B}\vec{u} \leq x_{max} - \mathcal{A} x_{0}
$$
and
$$
-(\mathcal{A} x_{0} + \mathcal{B}\vec{u}) 
\leq x_{min} \implies 
- \mathcal{B}\vec{u} 
- \leq -x_{min} + \mathcal{A} x_{0}
$$

Vectorizing the state constraints, we get
$$
G_{x} \ \vec{u} \leq h_{x}
$$
where
$$
G_{x} = \begin{bmatrix}
\mathcal{B} \\ -\mathcal{B}
\end{bmatrix} \in \mathbb{R}^{2N \times N}
, \quad h_{u} = \begin{bmatrix}
x_{max} - \mathcal{A} x_{0} \\  
-x_{min} + \mathcal{A} x_{0}
\end{bmatrix} \in \mathbb{R}^{2N}
$$

`Combining the Constraints`
Input control constraints and state constraints can be stacked together as
$$
G = \begin{bmatrix}
G_{u} \\ G_{x}
\end{bmatrix}
= \begin{bmatrix}
I_{N} \\ -I_{N} \\ \mathcal{B} \\ \mathcal{-B}
\end{bmatrix}
, \quad 
h = \begin{bmatrix}
h_{u} \\ h_{x}
\end{bmatrix}
= \begin{bmatrix}
\vec{u}_{max} \\
\vec{u}_{min} \\
x_{max} - \mathcal{A} x_{0} \\  
-x_{min} + \mathcal{A} x_{0}
\end{bmatrix}
$$

Hence, the optimization becomes
$$
\boxed{ \ \min_{u} \frac{1}{2} u^T H u + f^T u \ 
\quad s.t. \ Gu \leq h}
$$
where 
- $H = 2 \ (B^TQB + R)$ 
- $f = 2B^T \ QA \ x_{0}$

---
## See Also 
- [[MPC Controller]]
- [[Controller]]
