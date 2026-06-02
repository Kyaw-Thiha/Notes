# Sensor Model
#robotics/sensor/model 
A `sensor model` $r(x)$ is a function representing the [[Sensor|Sensor's]] response given a physical quantity measure $x$.

$$
r(x) = f(x)
$$
where
- $x$ is a physical quantity
- $r(x)$ is the sensor's response to $x$
- $f(x)$ is a simple function (`linear`/`logarithmic`) that represents the sensor response $r(x)$

---
`Simple Model`
The simplest model for sensors is an `affine model`
$$
r(x) = ax + b
$$
which allows us to recover value of $x$ by computing $x = \frac{r(x) - b}{a}$
This is also called `linear response function`.

---
## Sensing Process
The process of converting a physical quantity into values inside a computer involve multiple steps.

1. `Transduction`: Converting physical quantity into electric signals
2. `Sampling`: Converting continuous sensor readings into discrete digital format
3. `Discretization`: Clipping and quantizing the sampled data

[[Sensing Process|Read More]]

---
## Handling Noise
Note that every steps of the [[Sensing Process]] introduces noise and distortions.
Hence, we can denote as
$$
s(k) = r(k) + n(k)
$$
where
- $s(k)$ is `final sampled signal` value
- $r(k)$ is `correct signal` value (undistorted, noise-free) 
- $n(k)$ is the `noise term`

`Reasonable Assumptions`
- Noise values are `uncorrelated`.
  Noise component for $s(j)$ does not depends on $s(k)$
- Noise is `zero-mean`.
  Average of noise values should approach $0$.

> Do note that if you have information about sensor noise (for example through `sensor calibration`), then you should use it.

---
`Noise Removal`

Using the reasonable assumptions above, we have $2$ main methods for noise removal
1. `Averaging Multiple Readings`
2. `Local Smoothing (Linear Filtering / Convolution)`

---
`Denoising Categorical/Discrete Measurements`

Since neither the signal nor added value is not real-valued, averaging does not work.
Instead, we take the `median` /`mode` of the multiple readings.

---
## See Also
- [Paco's Notes](https://www.cs.utoronto.ca/~strider/docs/C85_Sensors.pdf)
- [[Sensor]]
- [[Sensing Process]]
- [[Sensor Noise]]
- [[Frequency Analysis]]
