# Inertial Navigation
#robotics/localization/inertial-navigation
`Inertial Navigation` performs dead-reckoning [[Localization]] by integrating sensor data to estimate robot's state.

![Inertial Navigation|300](https://timeandnavigation.si.edu/sites/default/files/multimedia-assets/500-si_fl_inertial_nav_18_fa_flat_rev_5-15.jpg)


---
## Components
Inertial Navigation mainly uses [[Accelerometer]] and [[Gyroscope]].

![Components of Inertial Navigation|400](https://europe1.discourse-cdn.com/arduino/original/4X/2/1/d/21de57f5a75b9d10a7513db8d07404882b294fc6.gif)

Three [[Accelerometer]] oriented orthogonally are needed for 3 axis of motion.
Likewise, three [[Gyroscope]] are needed for 3 axis of rotation.

---
### Computing Position and Orientation
From the acceleration data provided by sensor,
- `Orientation`: Integrating angle's rate of change over time
- `Velocity`: Integrating linear acceleration over time
- `Position`: Integrating velocity over time

---
## Coordinate Frames
To determine robot's orientation, we need two coordinate frames
- `Global Coordinate Frame`
  Describes position and orientation of robot as well as its environment $w.r.t$ predefined point chosen as origin
- `Body Coordinate Frame`
  Consists of central location (center of robot), and direction of coordinate axes aligned with robot's body

### Orientation

`Orientation Conversion`
For orientation conversion,
$$
\vec{g} = C \ . \ \vec{b}
$$
where
- $C$ is the $3 \times 3$ `rotation matrix`
- $\vec{g}$ is the `global coordinate frame`
- $\vec{b}$ is the `body coordinate frame`

We compute the rotation matrix $C$ by integrating the angle rate of change provided by gryo sensors.

`Updating rotation matrix`
Using the [derivation here](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-696.pdf), we get
$$
C(t + \delta t)
= C(t) + \left( I + \frac{\sin(\sigma)}{\sigma} B 
+ \frac{1 - \cos(\sigma)}{\sigma^2} B \right)
$$
where 
- $I$ is the $3 \times 3$ identity matrix
- $\sigma = |\vec{w}_{b} \ \delta t|$ where $\vec{w}_{b} = [w_{bx} \quad w_{by} \quad w_{bz}]$ is the the rate of change along body coordinate frame coordinate axes.
- $\delta t$ is the change in time
- 
$$
B = \begin{bmatrix}
0 & -w_{bz} \ \delta t & w_{by} \ \delta t \\
w_{bz} \ \delta t & 0 & -w_{bx} \ \delta t \\
-w_{by} \ \delta t & w_{bx} \ \delta t & 0
\end{bmatrix}
$$

### Position

`Estimating Position`
For converting the acceleration,
$$
\vec{a}_{g}(t) = C(t) \cdot \vec{a}_{b}(t)
$$
where
- $C(t)$ is the $3 \times 3$ `rotation matrix` at time $t$
- $\vec{a_{b}}(t) = [a_{bx}(t) \quad a_{by}(t) \quad a_{bz}(t)]$ is the `body coordinate frame`
- $\vec{a_{g}}(t)$ is the `global coordinate frame` at time $t$

Given the acceleration vector in global coordinate frame,
$$
\vec{v}_{g} (t + \delta t)
= \vec{v}_{g} + \theta \cdot 
(\vec{a} \ (t + \delta t) - \vec{g}_{g})
$$
Note that 
- We are using [[First-Order Taylor Expansion]]
- We are compensating for gravity

Given the velocity vector in global coordinate frame,
$$
\vec{s}_{g}(t + \delta t)
= \vec{s}_{g} (t) + \delta t \cdot \vec{v}_{g}(t + \delta t)
$$
we get the position vector.

---
`Drift`
Note that we have inaccuracies due to
- Noise is gyrometer and accelerometer
- Approximation from [[First-Order Taylor Expansion|Taylor's Approximation]]
- Other inaccuracies in integration process

This error introduced into the estimation process for orientation, velocity, and position is called `drift`.

This error can be significant over longer intervals of time.

`How is it useful?`
Given problem of `drift`, inertial navigation is not reliable over long intervals of time.

However for shorter intervals, it is useful for
- Situations [[Localization]] doesn't work $(\text{e.g lack of distinct landmarks})$
- [[GPS]] is unavailable $(\text{E.g: Urban Environment, Mars})$
- `Motion Compensation` $(\text{E.g: Drone compensating against wind})$

---
## See Also
- [Paco's Notes](https://www.cs.utoronto.ca/~strider/docs/C85_InertialNav.pdf)
- [[Accelerometer]]
- [[Gyroscope]]
- [[Localization]]
