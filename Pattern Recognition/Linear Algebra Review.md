`Einsum`
- $V^d = u_{d} = \sum_{r} u_{i}v_{i}$
- 
$$
\begin{align}
T^{ml}_{n}  
&= [M_{1n}^m, M_{2n}^m, \dots, M^m_{\ln}] \\[6pt]
&= [u^TM_{1}v, u^TM_{2}v, \dots, u^TM_{l}v]
\end{align}
$$

[[Einsum|Read More]]

---
`Properties of Matrices`
- $AB \neq BA$
- $A(B + C)$
- $A(BC) = (AB)C$

---
`Eigen`
Suppose $A^{-1}A = I$
$\exists A^{-1} \implies Ax = b$
Note that $Ax = \sum x_{i} A_{i} = \sum c_{i} \vec{v}_{1}$

---
`Norms`
Measuring the size of the vector.
$L^p$ Norms: $||\vec{x}||_{p} = \left( \sum_{i} |x_{i}|^p \right)^{1/p}$, $\forall p\in R, p\geq 1$

Here are some properties of Norm:
1. $f(\vec{x}) = 0 \implies \vec{x} = 0\vec{}$
2. $f(\vec{x} + \vec{y}) \leq f(\vec{x}) + f(\vec{y})$
3. $\forall \alpha\in R, \ f(\alpha \vec{x}) = |\alpha| \ f(\vec{x})$

Here are some common norms
- $L^1 \implies ||\vec{x}|| = \sum_{i} |x_{i}|$
- $L^2 \implies ||\vec{x}||_{2} = (\sum_{i} |x_{i}|^2)^{1/2}$
- $L^{\infty} \implies ||\vec{x}||_{\infty} = \max_{i} \ |x_{i}|$

---
`Frobenius Norm`
This is essentially a norm for a matrix.
$||A||_{F} = \sqrt{ \sum_{ij} A_{ij}^2 }$

$u.v = ||u||_{2} \ ||v||_{2} \cos(\theta)_{w}$

---
`Eigen Decomposition`
Let 
$$
A\vec{x} = \lambda \vec{x}
$$
where
- $\vec{x}$ is an eigenvector
- $\lambda$ is an eigenvalue

Then,
$$
A = V \ diag(\lambda) \ V^{-1}
$$

---
`Single Value Decomposition (SVD)`
It has the form
$$
A = U \ D \ V^T
$$
where
- $U$ is the `left singular matrix`
- $V$ is the `right singular matrix`
- $D$ is
$$
\begin{bmatrix}
\lambda_{1} & & & 0 & 0 \\
0 & \lambda_{2} & & 0 & 0\\
0 & 0 & \ddots & 0 & 0 \\
0 & 0 & \dots & \lambda_{n} & 0 & \dots & 0 \\
0 & 0 & \dots & 0 & 0 & \dots & 0 \\
0 & 0 & \dots & 0 & 0 & \ddots & 0 \\
0 & 0 & \dots & 0 & 0 & \dots & 0 \\
\end{bmatrix}
$$

[[Single Value Decomposition (SVD)|Read More]]

---
`Moore-Penrose Pseudo-Inverse`
Let 
$$
A \vec{x} = \vec{y}
$$
where $A$ is an uninvertible matrix.
Then,
$$
\begin{align}
A^TA \ \vec{x} &= A^T y \\[6pt]
\vec{x} &= (A^T A)^{-1} \ A^T y \\[6pt]
\vec{x} &= (A^T A + \alpha I)^{-1} \ A^T y \\[6pt]
\end{align}
$$
Next, we can use [[Single Value Decomposition (SVD)]] to get
- $A = UDV^T$
- $A^T = VD^TU^T$
	where 
$$
D^T = \begin{bmatrix}
\frac{1}{\alpha_{1}} \\
& \frac{1}{\alpha_{2}} \\
& & \ddots  \\
& & & \frac{1}{d_{n}} \\
& & & & 0  \\
& & & & & \ddots  \\
& & & & & & 0  \\
\end{bmatrix}
$$

[[Moore-Penrose Pseudoinverse|Read More]]

---
`Gradient Based Optimization`
Suppose we want to find
$$
\theta^* = \arg \min f(\theta)
$$
Given $f(\theta)$, we can optimize it by $\frac{d}{d\theta} f(\theta) = 0$.

We can get that
$$
\nabla_{\theta} f(\vec{\theta})
= \begin{bmatrix}
\frac{\partial f(\vec{\theta})}{\partial \theta_{1}} \\
\frac{\partial f(\vec{\theta})}{\partial \theta_{2}} \\
\vdots \\
\frac{\partial f(\vec{\theta})}{\partial \theta_{d}} \\
\end{bmatrix}
$$

For a multi-dimension $f: R^d \to R^m$, we can get the `Jacobian Matrix`
$$
J(f) = \begin{bmatrix}
\frac{\partial f_{1}}{\partial \theta_{1}}
& \frac{\partial f_{2}}{\partial \theta_{1}}
& \dots
& \frac{\partial f_{n}}{\partial \theta_{1}} \\
\vdots & \vdots & \ddots & \vdots  \\
\frac{\partial f_{1}}{\partial \theta_{d}} &
\frac{\partial f_{2}}{\partial \theta_{d}} 
& \dots
& \frac{\partial f_{n}}{\partial \theta_{d}} \\
\end{bmatrix}
$$

---
