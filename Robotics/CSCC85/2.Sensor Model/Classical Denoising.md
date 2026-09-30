# Classical Denoising

---
## Mode
We can take the mode over a window of observations. 
This allows us to ignore the anomalous sensor readings.

---
## Mean
We can take the mean over a window observations. This allow us to average out the sensor readings and cancel out anomalous readings.

---
## Gaussian Filter

---
## Exponential Moving Average(EMA)
[[#Exponential Moving Average(EMA)|EMA]] can be used to give more weight to the current observation, while the prior state can help smooth it out in case of anomalous readings.

$$
y[i] = \alpha x[i] + (1 - \alpha) \cdot y[i-1]
$$
where 
- $x[i]$ is the current observation at $i$
- $y[i]$ is the current output at $i$
- $\alpha$ is a weighing factor 

---
