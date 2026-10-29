# Inverse Velocity Kinematics
Given an [[Config Space(C-Space)|initial configuration]] $\theta^{0}$ and desired [[Twist|end-effector twist]] $\mathcal{V}_{d}$(linear and angular), we can determine
- feasible joint trajectory
- joint values $\theta$
- joint velocities $\dot{\theta}$

$$
\mathcal{V}_d = J(\theta)\dot{\theta} \quad \Longrightarrow \quad \dot{\theta} = J^{-1}(\theta)\mathcal{V}_d
$$
Note the challenges of $J^{-1}$ are
- near singularity
- non invertible $J$ (singular configuration or $n \neq m$)

---
## Naive Approach
A naive approach to [[Inverse Velocity Kinematics|inverse velocity kinematics]] would be
$$
\mathcal{V}_d = J(\theta^0)\dot{\theta} \quad \Longrightarrow \quad \dot{\theta} = J^{-1}(\theta^0)\mathcal{V}_d
$$
where
$$
\begin{align*}
\theta^1 &= \theta^0 + \dot{\theta} \\[0.5em]
\theta^{t+1} &= \theta^t + \dot{\theta}\,\Delta t
\end{align*}
$$

---
### Trying Out Naive Approach
> First we can choose the desired [[Velocity Kinematics|Cartesian velocity]] $\mathcal{V}_{d}$.
> Then, we convert to required [[Velocity Kinematics|joint velocity]] $\dot{\theta} = J^{-1}(\theta^{0}) \ \mathcal{V}^{d}$.

Given the following
$$
\theta^0 = \begin{bmatrix} 10^\circ \\ 30^\circ \end{bmatrix} \qquad\qquad \mathcal{V}_d = \begin{bmatrix} -0.1 \\ 0.1 \end{bmatrix}
$$
This gets us the velocity of 
$$
\dot{\theta} = J_v^{-1}(\theta^0)\,\mathcal{V}_d = \begin{bmatrix} -0.0107 \\ 0.0654 \end{bmatrix}
$$
and substituting it in, we get
$$
\theta^{t+1} = \theta^t + \dot{\theta}\,\Delta t \qquad\qquad \theta^{10} = \begin{bmatrix} 3.86^\circ \\ 67.45^\circ \end{bmatrix}
$$

> Note that as the joints move, the [[Robot Jacobian|Jacobian]] needs to be updated.

---
## Resolved-Rates Motion Control
For a given input $\mathcal{V}_{d}$, 
- Consider discrete time steps $1,\ 2,\ \dots$
- Update computation every $\Delta_{t}$
- At each time step: 
	- $(i)$ Compute [[Velocity Kinematics|joint velocity]] $\dot{\theta} = J^{-1}_{\mathcal{V}}(\theta) \ \mathcal{V}_{d}$.
	- $(ii)$ Compute next joint value $\theta^{t+1} = \theta^{t} + \Delta_{t} \ \dot{\theta}$.
	- $(iii)$ Move robot joints to $\theta^{t+1}$.

![image|500](https://notes-media.kthiha.com/Inverse-Velocity-Kinematics/769f48021e7d309a9618cfdb01e81010.png)

---
### Example
> First we can choose the desired [[Velocity Kinematics|Cartesian velocity]] $\mathcal{V}_{d}$.
> Then, we convert to required [[Velocity Kinematics|joint velocity]] $\dot{\theta} = J^{-1}(\theta^{0}) \ \mathcal{V}^{d}$.

Given the following
$$
\theta^0 = \begin{bmatrix} 10^\circ \\ 30^\circ \end{bmatrix} \qquad\qquad \mathcal{V}_d = \begin{bmatrix} -0.1 \\ 0.1 \end{bmatrix}
$$
This gets us the velocity of 
$$
\dot{\theta} = J_v^{-1}(\theta^{-1})\,\mathcal{V}_d 
$$
and substituting it in, we get
$$
\theta^{t+1} = \theta^t + \dot{\theta}^{t+1} 
\qquad\qquad 
\theta^{10} = \begin{bmatrix} -23.65^\circ \\ 98.81^\circ \end{bmatrix}
$$

---
## Damped Least-Squares (DLS) Inverse
[[#Damped Least-Squares (DLS) Inverse|DLS Inverse]] is the inversion of [[Velocity Kinematics|Differential Kinematics]] as an optimization problem.
- Follow desired motion: $||J \dot{\theta} - \mathcal{V}_{d} ||$ to minimize Cartesian velocity error
- Avoid excessive joint motion: $|| \dot{\theta} ||$ to minimize joint velocity magnitude

Combining both objectives into one optimization problem, we get
$$
\min_{\dot{\theta}} \; \frac{1}{2}\underbrace{\left\|J\dot{\theta} - \mathcal{V}_d\right\|}_{\bbox[#E2EFDA,5px]{\color{black}\text{tracking error}}} + 
\underbrace{\frac{k}{2} \ \left\|\dot{\theta}\right\|}_{\bbox[#DDEBF7,5px]{\color{black}\text{joint velocity damping}}}
$$
where $k \geq 0$ is the `damping factor` which controls the trade-off between tracking accuracy and joint velocity magnitude.

---
### Deriving the DLS Solution
Given the optimization objective
$$
\min_{\dot{\theta}} H = \bbox[#D9EAD3,5px]{\color{black}\frac{1}{2}\left\|J\dot{\theta} - \mathcal{V}_d\right\|} + \bbox[#CFE2F3,5px]{\color{black}\frac{k}{2}\left\|\dot{\theta}\right\|}
$$
differentiating it gets us
$$
\frac{\partial H}{\partial \dot{\theta}} = J^T\left(J\dot{\theta} - \mathcal{V}_d\right) + k\dot{\theta} = 0 \quad \Longrightarrow \quad \dot{\theta} = \underbrace{J^T\left(JJ^T + kI\right)^{-1}}_{\bbox[#F4CCCC,5px]{\color{black}J_{\text{DLS}} \;\; \text{Singularity Robust Inverse}}} \mathcal{V}_d
$$
where
- $k=0:$ ordinary least-squares solution $J_{DLS} = J^{\dagger} = J^{T}(JJ^{T})^{-1}$.
- $k>0:$ regularized solution that remains well-defined even at singularities.
- increasing $k:$ generally reduces joint velocity magnitude at the expense of tracking accuracy

---
