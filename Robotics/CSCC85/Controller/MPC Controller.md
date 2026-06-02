# MPC Controller
#robotics/controller/mpc 

`Model-Predictive-Control Controller` is a [[Controller]] that 
1. Calculates optimal control actions over a finite time horizon
2. Applying only first step, and re-optimizing repeatedly.
3. Has state and control input constraints

![MPC Controller|400](https://www.researchgate.net/profile/Md-Sohel-Rana-3/publication/262568366/figure/fig6/AS:670533607755787@1536879078812/Typical-structure-of-model-predictive-controller.png)

Compared to [[LQR Controller]], `MPC Controller` has a finite horizon, uses continuous online optimization, and has explicit constraints.

---
## System Model
`MPC Controller` uses a linear state system defined as
$$
x_{k+1} = Ax_{k} + Bu_{k}
$$
where
- $\vec{x}_{k}$ is the current `state vector` $(\text{position, velocity, angle, etc})$
- $\vec{x}_{k+1}$ is the next `state vector`
- $\vec{u}_{k}$ is the `control input` $(\text{force, torque, voltage})$
- $A$ and $B$ are `system matrices`

Note that compared to [[LQR Controller]], the states and control input are represented discretely.

---
## Finite-Horizon Optimal Control Problem

`Finite-Horizon Prediction`
At an arbitrary step $k$, `MPC` predicts state vectors $\{ x_{k}, x_{k+1}, \dots, x_{N} \}$, as well as their corresponding control inputs $\{ u_{k}, u_{k+1}, \dots, u_{N} \}$.

`Cost Function`
`MPC` minimizes the following cost function
$$
J = \sum^{N-1}_{k=0} 
(x_{k}^T Q x_{k} + u_{k}^T R u_{k})
+ x_{N}^T \ Q_{f} \ x_{N}
$$
where
- $\vec{x}$ is the `state vector`
- $\vec{u}$ is the `control input`
- $Q$ is the `error penalizing matrix`
- $R$ is the `aggressive control penalizing matrix`
- $Q_{f}$ is the `terminal error cost`

`Constraints`
In `MPC`, we can explicitly define 
- State Constraints:  $x_{min} \leq x_{k} \leq x_{max}$ 
  $(\ \text{E.g: Robot Joint Angle Limits, Drone Altitude Limits} \ )$
- Control Constraints: $u_{min} \leq u_{k} \leq u_{max}$ 
  $( \ \text{E.g: Robot Joint Torque Limits, Car Acceleration/Brake Limit} \ )$

---
## Optimization
This control problem can be derived as optimization of 
$$
\boxed{ \ \min_{u} \frac{1}{2} u^T H u + f^T u \ 
\quad s.t. \ Gu \leq h}
$$
where 
- $H = 2 \ (B^TQB + R)$ 
- $f = 2B^T \ QA \ x_{0}$

[[Maths behind MPC Controller|Read More]]

---
## See Also
- [[Controller]]
- [[LQR Controller]]
- [[Maths behind MPC Controller]]
