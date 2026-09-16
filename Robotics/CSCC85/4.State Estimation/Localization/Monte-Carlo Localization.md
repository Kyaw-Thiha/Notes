# Monte-Carlo Localization
#robotics/localization/particle-filter #robotics/localization/monte-carlo
$\text{Monte-Carlo Localization (MCL)}$ $[\text{Particle-Filter Localization}]$ is a [[Localization]] technique that represents uncertainty by using many particles that 
- are moved
- weighted by sensor data, and
- resampled to converge on true location.

![Monte-Carlo Localization|400](https://assets.omscs.io/notes/particle-filters.gif)

---
### Main Mechanism

The `particle-filter` runs a loop with 3 steps:

`Action`
- The robot carries out a random or predefined action.
- Each particles also update its state accordingly.
- But it must model noise by using a `motion model`.

`Sensing`
- Each particle has `ground truth value` $(\text{GT})$ 
- The robot obtain the sensor readings at the new pose.
- Each particle compute the error between $\text{GT}$ and $\text{Sensor Readings}$
- The error probability is computed using a `sensor model`
- Re-normalize the beliefs across all particles.

`Resampling`
- Sample particles with probability proportional to $Bel(x_{i})$
- To mitigate wrong convergence, a fixed percentage of particles $(\text{e.g: 5\%})$ is drawn uniformly.
- Re-normalize the beliefs across the chosen particles. 

---
### Mathematical Model
`Particle Filtering` can be thought of as [[Monte Carlo Methods|Monte Carlo]] approximation of [[Histogram Localization|Bayesian Localization]].

`State`
Suppose the state modelled is position and heading direction. Then, 
$$x_{i} = (x, y, \theta)$$

We want to get a categorical distribution
$$
Bel(x) = p(x = x_{i} \mid a_{1}, z_{1}, \ \dots, \ a_{k}, z_{k})
$$

Which can be approximated by
$$
Bel(x) \approx \sum^N_{i=1} w_{i} \ \delta(x-x_{i})
$$

`Motion Model`
The action can be represented as
$$
x_{i}^{(t)}
\sim p(x_{t} \mid x_{i}^{(t-1)}, \ a_{t})
$$

Adding in the noise,
$$
\begin{align}
d &= d_{original} + \mathcal{N}(0, \sigma^2) \\[6pt]
\theta &= \theta_{original} + \mathcal{N} (0, \sigma^2)
\end{align}
$$

`Sensor Model`
For each $\text{particle-}i$ and $\text{sensor ray-}j$,
$$
e_{ij} = GT_{ij} - z_{j}
$$

For each sensor ray $j$,
$$
p(err_{j}) 
= \frac{1}{\sqrt{ 2\pi \sigma^2 }}
\ \exp \left( \frac{-(err_{j} - \mu)^2}{2\sigma^2} \right)
$$
Assuming independence across sensor rays
$$
p(z | x_{i}) 
= \prod^w_{j=1} \frac{1}{\sqrt{ 2\pi \sigma^2 }}
\ \exp \left( \frac{-(err_{ij} - \mu)^2}{2\sigma^2} \right)
$$
which leads to 
$$
w_{i} 
= \prod^w_{j=1} \mathcal{N}(err_{ij};  0, \sigma^2)
$$

Hence, the belief per particle $i$ can be updated as
$$
Bel(x_{i}) \gets Bel(x_{i}) \ p(z \mid x_{i})
$$

`Resampling`
The particles are sampled proportional to their belief
$$
x_{i} \sim Bel(x)
$$
A fixed percentage of particles are drawn uniformly
$$
x_{i} \sim \text{Uniform}(0,1)
$$

`Note: Sensor Calibration`
Note that a sensor calibration can be used instead of gaussian model for the motion and sensor.

---
## Limitations
- `Rate of Convergence` can be slow.
  Especially if initial set of particles are small and closest initial particle does not agree with the robot.
- `False Convergence` can occur.
  Especially symmetrical environments with no distinct features
  Sampling some particles from uniform distribution alleviate this problem.

---
## See Also
- [[Localization]]
- [[Histogram Localization]]
