# Controller
#robotics/controller 

`Controller` is a software in robotics system that change the states of physical quantities in the environment towards a desired value and maintain it.

![Controller](https://scaron.info/figures/feedback_loop.png)

---
### Modelling the System

`State`
We can represent the state as
$$
\vec{s} = \begin{bmatrix}
x_{1}  \\
x_{2} \\
\vdots \\
x_{n}
\end{bmatrix}
, \quad
\text{where } x_{i} \text{ represents a quantity of the state}
$$

`Linear Model`
Modelling the system as a linear system, we can get
$$
\dot{\vec{s}} 
= \frac{d\vec{s}}{dt}
= A.\vec{s}
$$
where
- $\frac{d\vec{s}}{dt}$ is the change in state variables over time
- $A$ is the matrix for applying linear function to state variables

`Controller Model`
We can also add the controller affecting the state as
$$
\dot{\vec{s}} 
= \frac{d\vec{s}}{dt}
= A.\vec{s} + B.\vec{u}
$$
where
- $\vec{u}$ is the control signal
- $B$ is the matrix for applying linear function

---
### Car Example
Consider an example of the `cruise control`.

In most simple terms, consider the state system of the car travelling horizontally. We get
$$
\frac{d\vec{s}}{dt}
= \begin{bmatrix}
\frac{dx}{dt} \\ \frac{dv_{x}}{dt}
\end{bmatrix}
= \begin{bmatrix}
0 & 1 \\ 0 & 0 
\end{bmatrix}
. \begin{bmatrix}
x \\ v_{x}
\end{bmatrix}
$$

Note that the velocity is consider constant in the system.
To model a system with changing velocity, we get
$$
\frac{d\vec{s}}{dt} 
= \begin{bmatrix}
\frac{dx}{dt}  \\
\frac{dv_{x}}{dt} \\
\frac{da_{x}}{dt}
\end{bmatrix}
= \begin{bmatrix}
0 & 1 & 0 \\
0 & 0 & 1 \\
0 & 0 & 0
\end{bmatrix}
. \begin{bmatrix}
x \\ v_{x} \\ a_{x}
\end{bmatrix}
$$

Adding in the `negative feedback loop` of the controller, we get
$$
\frac{d\vec{s}}{dt} 
= \begin{bmatrix}
\frac{dx}{dt}  \\
\frac{dv_{x}}{dt} \\
\frac{da_{x}}{dt}
\end{bmatrix}
= \begin{bmatrix}
0 & 1 & 0 \\
0 & 0 & 1 \\
0 & 0 & 0
\end{bmatrix}
. \begin{bmatrix}
x \\ v_{x} \\ a_{x}
\end{bmatrix}
+ \begin{bmatrix}
0 & 0 \\
0 & 0 \\
1 & -1
\end{bmatrix}
. \begin{bmatrix}
\text{accelerate} \\ \text{brake}
\end{bmatrix}
$$

Hence, the controller effects a change in state variables by changing the acceleration of the car $(\text{matrix } B)$.
This in turn changes the velocity, and then the position of the car $(\text{matrix } A)$.

---
### Controller Implementations

`Bang-Bang Controller`
This is the simplest `controller` which try to push the state towards the desired reference value.
[[Bang-Bang Controller|Read More]]


`PID Controller`
This is a controller that uses the `error`, the `integral over past error`, and the `change in error`.
[[PID Controller|Read More]]

`LQR Controller`
This is the controller that optimizes a quadratic cost function to balances state error and control effort.
[[LQR Controller|Read More]]

`MPC Controller`
This is the controller that optimizes a quadratic cost function 
- over a finite horizon 
- has explicit state and control constraints, and 
- recomputed in each step

[[MPC Controller|Read More]]

---
## See Also
- [Paco's Notes](https://www.cs.utoronto.ca/~strider/docs/C85_Controllers.pdf)
- [[Bang-Bang Controller]]
- [[PID Controller]]
- [[LQR Controller]]
- [[MPC Controller]]
