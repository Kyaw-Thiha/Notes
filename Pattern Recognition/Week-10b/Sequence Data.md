# Sequence Data
We can do the same things as would be done to any dataset.
- Classify it
- Regress against it
	- Module deep brain stimulation based on motion observations
	- Based on recent history, predict next item in the sequence
$$
x_{t+1} = f(x_{t}, \ \dots, \ x_{t-h})
$$
	- Based on recent history, predict series of items.
$$
(x_{t+p}\ , \ \dots, \ x_{t+1}) = f(x_{t} \ , \ \dots, \ x_{t-h})
$$

---
## Auto-Regressive Models
An $\text{AR}(p)$ uses the last $p$ observations to predict the next observation:
$$
\hat{X}
$$