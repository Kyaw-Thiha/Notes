# RNN
[[Recurrent Neural Network (RNN)|RNN]] is a type of [[Neural Network|neural network]] that process sequential and time series by using internal memory.

![image|400](https://notes-media.kthiha.com/Recurrent-Neural-Network-(RNN)/5e29baec6b8e36a71f2bd7fa9b37289a.png)

---
## Mechanism

![400](https://cdn.analyticsvidhya.com/wp-content/uploads/2024/09/image-80.webp)

The previous hidden state $h_{t-1}$ is used in the hidden state $h_{t}$.

$$
\begin{align}
&h_{t} = \sigma_{h}(W_{h} x_{t} + U_{h}h_{t-1}  
+ b_{h}) \\[6pt]
&y_{t} = \sigma_{y}(W_{y}h_{t} + b_{y})
\end{align}
$$

---
## Weakness
Consider the [[Recurrent Neural Network (RNN)|RNNs]] unrolled onto a long sequence. They become very deep since 
$$
\text{Depth} = \text{Length of Sequence}
$$
This causes 2 problems:
- not good at modelling long-term dependencies
- hard to train due to [[Vanishing and Exploding Gradients|vanishing/exploding gradients]]

---
### Exploding/Vanishing Gradient
Suppose the update function is a simple linear model with inputs ignored:
$$
h_{t} = W_{h} h_{t-1}
$$
We can write this for all time steps as
$$
h_{t} = (W_{h})^{t} h_{0}
$$
Then, we would have
- exploding gradients $h_{t} \to \infty \quad \text{if} \quad |W_{h}| > 1$
- vanishing gradients $h_{t} \to 0 \quad \text{if} \quad |W_{h}| < 1$

---
#### Tackling Exploding Gradient
> Use **gradient clipping**. 
> If the gradient is greater than a threshold, set gradient to threshold.  

---
#### Tackling Vanishing Gradient
Use [[Skip Connections|skip connections]] to ensure hidden state over the long term.
![image|400](https://notes-media.kthiha.com/Recurrent-Neural-Network-(RNN)/0d1d3ca653583025333fdff6a074fdd3.png)

---
## See Also
- [[Recurrent Neural Network (RNN)]]