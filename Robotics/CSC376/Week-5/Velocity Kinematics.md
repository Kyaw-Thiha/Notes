# Velocity Kinematics
## Forward vs Velocity Kinematics
### Forward Kinematics
> We are finding the position and orientation of the end-effector frame $\{ b \}$ for a given set of joint positions $\theta$.

![image|200](https://notes-media.kthiha.com/Velocity-Kinematics/76d6987ff3e8e19edab68d41b9073ab0.png)

It can be represented as
$$
x(t) = f\big( \theta(t) \big)
$$

---
### Velocity Kinematics
> Spatial velocity of end-effector from a given set of joint positions $\theta$ and joint velocities $\dot{\theta}$.

![image|200](https://notes-media.kthiha.com/Velocity-Kinematics/665c724e285dcf2c082a26b0d67e43bf.png)

It can be represented as
$$
\dot{x} = \frac{\partial f(\theta)}{\partial \theta} \frac{d\theta(t)}{dt}
= \frac{\partial f(\theta)}{\partial \theta} \dot{\theta}
= J(\theta) \ \dot{\theta}
$$
where $J(\theta)$ is the [[Robot Jacobian|Robot Jacobian]].

---
