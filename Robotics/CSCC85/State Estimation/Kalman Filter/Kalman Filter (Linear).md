# Kalman Filter
#kalman-filter/linear 

`Kalman Filter` uses a system model to predict state variables while accounting for noise in sensing & actions.

![Kalman Filter|400](https://wirelesspi.com/wp-content/uploads/2024/05/figure-kalman-filter-conventional-algorithm.png)

---
## Assumptions 
- The system is represented by a `linear model`.
  Includes the noise term.
- Relation between state variables to sensor model is represented by a `linear model`.
  Includes a noise term.
- Noise in both system and sensor readings is [[Gaussian Distribution|zero-mean Gaussian]].

---
## Key Summaries
- `Kalman Filter` is a powerful estimator for state variables in a dynamic system.
- Relies on a `linear model` to predict state variable values
- Performs an adjustment based on `sensor measurments`
- This adjustment takes into account the `uncertainties` in the system and the sensor measurements.
---
## Model

`System Model`
The system model is given by
$$
\vec{x}_{t} 
= A_{t} \ \vec{x}_{t-1}
+ B_{t} \ \vec{u}_{t} + \vec{w}_{t}
\ , \quad \vec{w}_{t} \sim 
\mathcal{N}(\vec{\mu}_{w_{t}}, \ Q_{t})
$$
where
- $\vec{x}_{t}$ is the `state variables` vector
- $\vec{u}_{t}$ represents any `control inputs` to the system
- $\vec{w}_{t}$ represents `zero-mean Gaussian` with covariance $Q$
- $A$ is the `state transition matrix`
- $B$ relates `control inputs matrix`

---
`Sensor Model`
The linear sensor model is given by
$$
\vec{z}_{t}
= H_{t} \ \vec{x}_{t} + \vec{v}_{t}
\ , \quad \vec{v}_{t} \sim
\mathcal{N}(\vec{\mu}_{v_{t}}, \ R_{t})
$$
where
- $\vec{x}_{t}$ is the `state variables` vector
- $\vec{z}_{t}$ is the `sensor readings`
- $\vec{v}_{t}$ is the `zero-mean Gaussian` noise with covariance $R$
- $H_{t}$ is the mapping between `state to sensor readings`

---
`Process Covariance`
This `Process Covariance matrix` $P$ describes the uncertainty in the state variable estimates, and changes over time.

- Each update is affected by random noise
- Noise at $t=c$ will be factored into every update for $t>c$.
- Hence, noise compounds over time.
- Therefore if we do nothing other than apply our `linear model`, then uncertainty should grows over time.

---
## Algorithm
The `Kalman Filter` consists of two steps: 
- Prediction
- Measurement Update

`Prediction`
The prediction step is denoted as
$$
\begin{align}
\check{x}_{t}
&= A_{t} \ \hat{x}_{t-1} + B_{t} \ \vec{u}_{t}
& & (1) \\[6pt]

\check{P}_{t} &= A_{t} \ \hat{P}_{t-1} \ A_{t}^T
+ Q_{t} & & (2)
\end{align}
$$
where
- $\check{x}_{t}$ is `predicted state variables` of the first $(\text{prediction})$ step
- $\hat{x}_{t-1}$ is the `final estimate` of state variables after completing both steps $(\text{prediction and measurement update})$ of the Kalman Filter in the previous timestep $t-1$
- $\vec{u}_{t}$ is the `control input`

and
- $\check{P}_{t}$ is the `predicted process covariance` of the first $(\text{prediction})$ step
- $\hat{P}_{t-1}$ is the `final estimate of process covariance` after completing both steps $\text{(prediction and measurement update)}$ in the previous timestep $t-1$
- $Q_{t}$ is the `Gaussian Covariance` of the system noise

---
`Prediction Notes`

- Note that predicted state variables at time $t$ uses the `final Kalman estimate` of state variables at time $t-1$.
  The same goes for process covariance too.

- $\text{Equation-(1)}$ simply applies the linear system to the previous Kalman estimate for the `state variables`, without the noise term.

- Likewise, $\text{Equation-(2)}$ applies the linear system to the previous Kalman estimate of `uncertainty`.
  But this time, it also injects new uncertainty in terms of $Q_{t}$.

Note that 
$$
\begin{align}
&y = Ax \\[6pt]
\implies &Cov(y) = APA^T
\end{align}
$$

---
`Measurement Update`
Given $\text{step-1}$ predictions, KF uses sensor measurements the update the estimates of state variables.

The measurement update step is denoted as
$$
\begin{align}
\hat{x}_{t}
&= \check{x}_{t}  
+ K_{t}(\vec{z}_{t} - H_{t} \ \check{x}_{t}) 
& & \quad (3) \\[6pt]

\hat{P}_{t} 
&=  \check{P}_{t} - K_{t} \ H_{t} \ \check{P}_{t}
& & \quad (4) \\[6pt]
\end{align}
$$
where
- $\hat{x}_{t}$ is the `final estimate` of state variables after completing both steps $(\text{prediction and measurement update})$ of the Kalman Filter at time $t$
- $\check{x}_{t}$ is `predicted state variables` of the first $(\text{prediction})$ step
- $\vec{z}_{t}$ is the `observed sensor measurement` at time $t$
- $H_{t}$ is the mapping between `state to sensor readings`
- $K_{t}$ is the `Kalman gain`

and
- $\hat{P}_{t}$ is the `final estimate of process covariance` after completing both steps$\text{(prediction and measurement update)}$ at time $t$
- $\check{P}_{t}$ is the `predicted process covariance` of the first $(\text{prediction})$ step

---
`Measurement Update Notes`
In $\text{Equation-(3)}$, $(\vec{z}_{t} - H_{t} \ \check{x}_{t})$ is comparing the difference between 
- sensor measurements $\vec{z}_{t}$ obtained at time $t$
- and predicted sensor measurements $\check{x}_{t}$ from $\text{Equation-(1)}$ with $H_{t}$ from our sensor model

The difference is then scaled by `Kalman gain` $K_{t}$.

This means that
- if difference between our predicted state variables $\check{x}_{t}$ and corresponding sensor measurement $\vec{z}_{t}$ is small, adjustment will be small.
- conversely if difference between our predicted state variables $\check{x}_{t}$ and corresponding sensor measurement $\vec{z}_{t}$ is large, adjustment will be large.

In $\text{Equation-(4)}$, the measurement update reduces the `uncertainty` in our process.

---
`Kalman Gain Update`
The Kalman Gain Update is given by
$$
K_{t} 
= \check{P}_{t} H_{t}^T
\ (H_{t} \check{P}_{t} H_{t}^T + R_{t})^{-1}
$$
where
- $\check{P}_{t}$ is the `predicted process covariance` of the first $(\text{prediction})$ step
- $H_{t}$ is the mapping between `state to sensor readings`
- $R_{t}$ is the covariance of zero-mean Gaussian sensor noise

This is derived by optimizing over the $(\text{error covariances of the Kalman estimate})^2$ in order to get the estimate closest to the true state variable value.

---
## Limitations
Relies on assumptions of 
- zero-mean Gaussian Noise 
- a linear system

If bad performances is encountered due to non-linearity, the `Extended Kalman Filter(EKF)` can be used.

---
## See Also
- [Paco's Notes](https://www.cs.utoronto.ca/~strider/docs/C85_KalmanFilters.pdf)
- [[Kalman Filter]]