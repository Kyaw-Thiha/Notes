# Sensing Process
#robotics/sensor/process 
The process of converting a physical quantity into values inside a computer involve multiple steps.

1. `Transduction`: Converting physical quantity into electric signals
2. `Sampling`: Converting continuous sensor readings into discrete digital format
3. `Discretization`: Clipping and quantizing the sampled data

Note that each of these steps attribute to [[Sensor Noise]].

---
### 1. Transduction
`Transduction` is converting physical quantity sensed by the sensor to an electrical signal.

The sensors are noisy so, the response of the sensor will not be an exact function of the input signal.
Sources of these noise include:
- `Sensor Limitations`
  Weak signals are not picked up by sensor
  Strong signals saturates the sensor (does not increase)
- `Sensor Non-Linearity`
- `Ambient Noise`
- `Electric, Thermal and RF Noise`

---
### 2. Sampling
`Sampling` is taking the value of the response $r(t)$ at uniformly spaced intervals.

The sensor response $r(t)$ is not usable by computer system since it is continuous and real-valued, so `sampling` is used.

$$
F( r(t) ) = s(k)
$$
where
- $F$ is the sampling function
- $r(t)$ is the continuous input signal
- $s(k)$ is the sampled signal

`Sampling Frequency` 
No. of readings by the sensor in one second.
Sampling frequency of $n$ $Hz$ means that the sensor would read $n$ times for each second.

`Sampling Period` 
The reading interval of the sensor response.
It has to be small enough to capture the fastest change that can occur in the signal.

`Aliasing`
Aliasing is when sampling frequency is slower than the fastest change in signal.
Thus, high frequency signal is misinterpreted as low frequency reading.
![Aliasing|400](https://docs-be.ni.com/bundle/labwindows-cvi/page/advancedanalysisconcepts/guid-932173fe-d89c-4bc0-a3ed-980f99bcc06b-help-web.png?_LANG=enus)

[[Frequency Analysis]] can be carried out to get minimum `sampling frequency` required to prevent `aliasing`.

---
### 3. Discretization 
`Discretization` is converting sampled signal into a useful range.

1. The sampled signal $s(k)$ has to be clipped/normalized into pre-defined range. (E.g: [0 to 255] for color)

2. `Quantization` is applied to round off sampled signal $s(k)$ to values of finite precision.
   We need this because integers and [[Floating Points]] inside computers have finite precision.

![Quantization|400](https://www.tutorialspoint.com/digital_communication/images/quantization.jpg)

---
## See Also
- [[Sensor Model]]
- [[Sensor]]
- [[Sensor Noise]]
- [[Frequency Analysis]]