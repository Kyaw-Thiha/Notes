# Time Scaling
[[Time Scaling Trajectory|Time scaling]] can be used to convert a [[Robotics Trajectory Generation|path]] into a [[Robotics Trajectory Generation|trajectory]] 
$$
\begin{array}{rl}
\textbf{Path} & \theta(s) = \theta_{\text{start}} + s\left(\theta_{\text{end}} - \theta_{\text{start}}\right), \quad s \in [0,1] \\[1em]
\textbf{Trajectory} & \theta\big(s(t)\big), \; \underbrace{s : [0,T] \to [0,1]}_{\textit{Time Scaling}}
\end{array}
$$

where we have
$$
\begin{array}{ccc}
\dot{\theta} = \dfrac{d\theta}{ds}\dot{s} & \hspace{4em} & \ddot{\theta} = \dfrac{d\theta}{ds}\ddot{s} + \dfrac{d^2\theta}{ds^2}\dot{s}^2 \\[1em]
\textit{Velocity} & & \textit{Acceleration}
\end{array}
$$

in which $\theta(s)$ and $s(t)$ must be twice differentiable.

---
## Time Scaling Techniques
We can [[Time Scaling Trajectory|time-scale]] [[Robotics Trajectory Generation|trajectories]] using one of
- [[#Third-Order Polynomial Time Scaling]]
- [[#Fifth-Order Polynomial Time Scaling]]
- [[#Trapezoidal Motion Profile]]
- [[#S-Curve Time Scaling]]

---
## Third-Order Polynomial Time Scaling
We can approximate the [[Robotics Trajectory Generation|trajectory]] using [[#Third-Order Polynomial Time Scaling|third order polynomials]] to get
$$
\begin{align*}
s(t) &= a_0 + a_1 t + a_2 t^2 + a_3 t^3, \qquad t \in [0, T] \\[0.7em]
\dot{s}(t) &= a_1 + 2a_2 t + 3a_3 t^2
\end{align*}
$$
and given the following constraints
$$
\begin{array}{lcc}
\textbf{Initial Constraints} & s(0) = 0 & \dot{s}(0) = 0 \\[0.7em]
\textbf{Terminal Constraints} & s(T) = 1 & \dot{s}(T) = 0
\end{array}
$$
we can get the following coefficients 
$$
\textbf{Coefficients} \qquad a_0 = 0, \quad a_1 = 0, \quad a_2 = \frac{3}{T^2}, \quad a_3 = -\frac{2}{T^3}
$$
which are obtained by evaluating $s(t)$ and $\dot{s}(t)$ at $t=0$ and $t=T$ and solving for $a_{0}, \dots, a_{3}$.

---
### Plots
Substituting in the coefficients, we get
$$
s(t) = \frac{3t^2}{T^2} - \frac{2t^3}{T^3}, \qquad t \in [0, T]
$$
which we can use to plot out

![image|400](https://notes-media.kthiha.com/Time-Scaling-Trajectory/4cd8b4f9e484811195fe33f397ebcd91.png)

---
### Evaluating the Trajectory
First, we can substitute into the straight line formula
$$
s(t) = \frac{3t^2}{T^2} - \frac{2t^3}{T^3}, \quad t \in [0, T] \quad 
\text{substitute into} \quad \theta(s) = \theta_{\text{start}} + 
s\left(\theta_{\text{end}} - \theta_{\text{start}}\right), \ s \in [0,1]
$$

Note that we need to choose the feasible $T$ such that 
$$
\begin{align*}
\left|\dot{\theta}\right| &\le \dot{\theta}_{\text{limit}} \\[0.5em]
\left|\ddot{\theta}\right| &\le \ddot{\theta}_{\text{limit}}
\end{align*}
$$

---
#### Displacement
We get the displacement of the [[Robotics Trajectory Generation|trajectory]] as 
$$
\theta(t) = \theta_{\text{start}} + \left(\frac{3t^2}{T^2} - \frac{2t^3}{T^3}\right)\left(\theta_{\text{end}} - \theta_{\text{start}}\right) \qquad t \in [0, T]
$$

---
#### Velocity
We get the velocity of the [[Robotics Trajectory Generation|trajectory]] as
$$
\dot{\theta}(t) = \left(\frac{6t}{T^2} - \frac{6t^2}{T^3}\right)\left(\theta_{\text{end}} - \theta_{\text{start}}\right) \qquad\qquad \dot{\theta}_{\max} = \frac{3}{2T}\left(\theta_{\text{end}} - \theta_{\text{start}}\right) \quad t = \frac{T}{2}
$$

---
#### Acceleration
We get the acceleration of the [[Robotics Trajectory Generation|trajectory]] as
$$
\begin{align*}
\ddot{\theta}(t) &= \left(\frac{6}{T^2} - \frac{12t}{T^3}\right)\left(\theta_{\text{end}} - \theta_{\text{start}}\right) \qquad \ddot{\theta}_{\max} = \left|\frac{6}{T^2}\left(\theta_{\text{end}} - \theta_{\text{start}}\right)\right| \qquad t = 0 \\[0.7em]
& \hspace{15.5em} \ddot{\theta}_{\min} = -\left|\frac{6}{T^2}\left(\theta_{\text{end}} - \theta_{\text{start}}\right)\right| \qquad t = T
\end{align*}
$$

---
## Fifth-Order Polynomial Time Scaling
We can approximate the [[Robotics Trajectory Generation|trajectory]] using [[#Fifth-Order Polynomial Time Scaling|fifth-order polynomial]] to get
$$
\begin{align*}
s(t) &= a_0 + a_1 t + a_2 t^2 + a_3 t^3 + a_4 t^4 + a_5 t^5, \qquad t \in [0, T] 
\\[0.7em]
\dot{s}(t) &= a_1 + 2a_2 t + 3a_3 t^2 + 4a_4 t^3 + 5a_5 t^4 \\[0.7em]
\ddot{s}(t) &= 2a_2 + 6a_3 t + 12a_4 t^2 + 20a_5 t^3
\end{align*}
$$
and given the following constraints
$$
\begin{array}{lccc}
\textbf{Initial Constraints} & s(0) = 0 & \dot{s}(0) = 0 & \ddot{s}(0) = 0 \\[0.7em]
\textbf{Terminal Constraints} & s(T) = 1 & \dot{s}(T) = 0 & \ddot{s}(T) = 0
\end{array}
$$

we can use these six constraints to solve uniquely for the $6$ coefficients.

---
### Plot
We can substitute coefficients into the formula of
$$
s(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3 + a_4 t^4 + a_5 t^5, \qquad t \in [0, T] 
$$

which we can use to plot out

![image|400](https://notes-media.kthiha.com/Time-Scaling-Trajectory/8c29b6309ce800d7652c5a1d8e45a148.png)

---
## Trapezoidal Motion Profile
We can approximate the [[Robotics Trajectory Generation|trajectory]] using [[#Trapezoidal Motion Profile|trapezoidal motion]] as
$$
\begin{array}{lcl}
t \in [0, t_a) & \hspace{4em} & \ddot{s} = a \\[0.7em]
t \in [t_a, T - t_a) & & \dot{s} = v \\[0.7em]
t \in [T - t_a, T] & & \ddot{s} = -a
\end{array}
$$
which can be visualized as

![image|400](https://notes-media.kthiha.com/Time-Scaling-Trajectory/03d2fa14a7c80755b24b1f77b6ac3657.png)

where two of $v, a$ and $T$ can be chosen independently.

---
### Fastest Straight Line Motion
The motion is not as smooth as [[#Third-Order Polynomial Time Scaling|cubic time scaling]], but largest $v$ and $a$ satisfying
$$
\begin{align*}
\left|\left(\theta_{\text{end}} - \theta_{\text{start}}\right) v\right| &\le \dot{\theta}_{\text{limit}} \\[0.7em]
\left|\left(\theta_{\text{end}} - \theta_{\text{start}}\right) a\right| &\le \ddot{\theta}_{\text{limit}}
\end{align*}
$$
is the largest straight-line motion possible, which can be visualized as

![image|200](https://notes-media.kthiha.com/Time-Scaling-Trajectory/c3a930be446385a274ad32e44ab1f600.png)

---
### Bang-Bang Motion
Note that the maximum velocity $v_{max}$ is never reached during motion $\frac{v^{2}}{a} > 1$.

![image|200](https://notes-media.kthiha.com/Time-Scaling-Trajectory/8be54c035b6f29467006fffaa19de260.png)

---
## S-Curve Time Scaling
> **S-Curve**
> $(i)$ Constant positive jerk until desired acceleration $\ddot{s}=a$ is achieved.
> $(ii)$ Constant acceleration until desired $\dot{s} = v$ is approached.
> $(iii)$ Constant negative jerk until $\ddot{s}=0$ at the time $\dot{s} = v$.
> $(iv)$ Constant velocity coasting at constant $v$.
> $(v)$ Constant negative jerk.
> $(vi)$ Constant deceleration.
> $(vii)$ Constant positive jerk until $\ddot{s}$ and $\dot{s}$ reach $0$ exactly at time $s=1$.

This can be visualized as

![image|400](https://notes-media.kthiha.com/Time-Scaling-Trajectory/ca2537243173e22287cbf18b9697055f.png)

---
## Polynomial via Point Trajectories
We can describe the [[Robotics Trajectory Generation|path]] with $\text{number of poses} > 2$ for more flexibility in designing the shape of the path. 

![image|300](https://notes-media.kthiha.com/Time-Scaling-Trajectory/51eb9af6a1155e6e018b39b12a74f4de.png)

We can assign intermediate points between the start and end to for example
- define lift off and settle down an object in pick-and-place tasks
- avoid obstacles during motion.

---
### Third-Order Polynomial Interpolation
#### Specified via Times and Velocities
Between via points $j$ and $j+1$ for joint $i$
$$
\theta_i(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3 \qquad t \in \left[T_j, T_{j+1}\right]
$$
where each segment between via points, for each joints, has $4$ coefficients and $4$ terminal conditions (initial and final position, and initial and final velocity).

This can be plotted as

![image|500](https://notes-media.kthiha.com/Time-Scaling-Trajectory/0819edbc7e13b3a238bf2b876d5f2964.png)

Note that velocities at via points are free, but velocity as well as acceleration before and after via point are constrained to be equal.

---
