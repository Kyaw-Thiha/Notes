# Backpropagation
#ml/models/neural-network/backpropagation 

`Backpropagation` is the main algorithm that enables the [[Neural Network]] to learn.

![Backpropagation](https://miro.medium.com/v2/resize:fit:1280/0*tSkOl1glLC8K79D-.gif)

---
## Prerequisite

`Loss Function`

Using notations from [[Vectorizing Neural Network]], we can define the [[Loss Function]] as
$$
L \left( y_{k}, \hat{y}_{k} \ ; \{ \tilde{W}^{(l)}_{l=1:L} \} \right) 
= L_{k} \left( \tilde{W}^{(l)}_{l=1:L} \right)
$$
where
- $y_{k}$ is the `target label` of data $k$
- $\hat{y}_{k}$ is the `predicted output`
- $\tilde{W}^{(l)}$ is the `augmented weight matrix` for layer $l$
- $\{ \tilde{W}^{(l)} \}_{l=1:L}$ is the set of all `parameters` of the network

`Cost Function`
Accordingly, the [[Loss Function#Cost Function|Cost Function]] can be defined as
$$
L \left( y_{k}, \hat{y}_{k} \ ; \{ \tilde{W}^{(l)}_{l=1:L} \} \right) 
= c \sum^N_{k=1} L_{k} 
\left( \{ \tilde{W}^{(l)} \}_{l=1:K} \right)
$$
where $c$ is usually $\frac{1}{N}$.

`Gradient Descent`
Recall from [[Gradient Descent]] that 
$$
w_{j,i}^{(l)}  
= w_{j,i}^{(l)} - \lambda \frac{\partial E}{\partial w_{j,i}^{(l)}}
$$
where $\frac{\partial E}{\partial w_{j,i}^{(l)}} = \sum^N_{k=1} \frac{\partial L_{k}(\{ W^{(l)} \}_{l=1:L} )}{\partial w_{j,i}^{(l)}}$

---
## Math

Consider the following [[Neural Network]].

![[Backpropagation.png|500]]

`Notations`
Similar to [[Vectorizing Neural Network]], let
- $z_{i}$ be the `pre-activation state` ($z_{i} = \sum^{m_{l}}_{l=1} (w_{i,l} \ a_{l}) + b_{l}$)
- $a_{i}$ be the `hidden state` ($a_{i} = \sigma( z_{i})$)
- $w_{i,l}$ be the `weight` connecting neuron $l$ from previous layer to neuron $i$ from current layer

`Loss Gradient`
Compute the loss gradient $w.r.t$ weights 
$$
\frac{\partial L}{\partial w_{i,l}}
= \underbrace{\frac{\partial L}{\partial z_{i}}}
_{\delta_{i}}
\times \underbrace{\frac{\partial z_{i}}{\partial w_{i,l}}}_{a_{l}}
$$

Compute the loss gradient $w.r.t$ bias 
$$
\begin{align}
\frac{\partial L}{\partial b^{(k)}}
&= \frac{\partial L}{\partial z^{(k)}}  
\odot \frac{\partial z_{i}^{(k)}}{\partial b^{(k)}} 
\\[6pt]

&= \frac{\partial L}{\partial z^{(k)}}  
\odot \frac{\partial \left(W^{(k)} \ a^{(k-1)} + b^{(k)} \right)} 
{\partial b^{(k)}}
\\[6pt]

&= \frac{\partial L}{\partial z^{(k)}}  
\odot 1
\\[6pt]

&= \underbrace{\frac{\partial L}{\partial z^{(k)}}}_{\delta^{(k)}}  
\\[6pt]
\end{align}
$$

`Weight Gradient`
Compute the pre-activation gradient of the current layer.
$$
\begin{align}
\frac{\partial z_{i}}{\partial w_{i,l}}
&= \frac{\partial \sum^{m_{l}}_{l=1} (w_{i,l} \ a_{l}) + b_{l}}{\partial w_{i,l}} \\[6pt]

&= \frac{\partial \sum^{m_{l}}_{l=1} w_{i,l} \ a_{l}}{\partial w_{i,l}} \\[6pt]

&= a_{l}
\end{align}
$$

`Preliminary Gradients`
The `pre-activation gradient` with respect to activation from previous layer can be denoted as
$$
\begin{align}
\frac{\partial z_{i}}{\partial a_{l}}  
&= \frac{\partial w_{i,l} \ a_{l}}{\partial a_{l}} \\[6pt]
&= w_{i,l}
\end{align}
$$

The `activation gradient` with respect to `pre-activation state` can be denoted as
$$
\begin{align}
\frac{\partial a_{i}}{\partial z_{i}}
&= \frac{\partial \sigma(z_{i})}{\partial z_{i}} \\[6pt]
&= \sigma'(z_{i})
\end{align}
$$

`Activation Gradient`
Compute the activation gradient of the current layer
$$
\begin{align}
\delta_{i}
&= \frac{\partial L}{\partial z_{i}} \\[6pt]

&= \sum^{m_{j}}_{l=1} \frac{\partial L}{\partial z_{j}} \times \frac{\partial z_{j}}{\partial z_{i}} \\[6pt]

&= \sum^{m_{j}}_{l=1}  
\underbrace{\frac{\partial L}{\partial z_{j}}}_{\delta_{j}}  
\times \underbrace{\frac{\partial z_{j}}{\partial a_{i}}}_{w_{j,i}} 
\times \underbrace{\frac{\partial a_{i}}{\partial z_{i}}}_{\sigma'(z_{i})} \\[6pt]

&= \sigma'(z_{i}) \sum^{m_{j}}_{j=1} \delta_{j} w_{j,i}
\end{align}
$$

Hence for an arbitrary [[Neural Network]] layer $k$,
$$
\delta_{i}^{(k)}
= \sigma'(z_{i}^{(k)}) \sum^{m_{k+1}}_{j=1} \delta_{j}^{(k+1)} w_{j,i}^{(k+1)}
$$

---
## Vectorizing
In real algorithms, we vectorize these for efficient computation.
Let
- $K$ be the number of neurons in layer $k-1$
- $N$ be the number of neurons in layer $k$
- $M$ be the number of neurons in layer $k+1$

`Vectorizing Weight Gradient`
The `hidden states` from previous layer $k-1$ can be denoted as
$$
a^{(k-1)} = 
\begin{bmatrix}
a^{(1)} \\[6pt]
a^{(2)} \\[6pt]
\vdots \\[6pt]
a^{(K)} \\[6pt]
\end{bmatrix}
$$

`Vectorizing Activation Gradient`
The `weights` connect current layer $k$ to the next layer $k+1$ can be denoted as
$$
W_{M \times N} = \begin{bmatrix}
w_{11} & w_{12} & \dots &  w_{1N} \\[6pt]
w_{21} & w_{22} & \dots & w_{2N} \\[6pt]
\vdots & \vdots &  & \vdots \\[6pt]
w_{M1} & w_{M2} & \dots & w_{MN} \\[6pt]
\end{bmatrix}
$$
Accordingly, we can denote
$$
\delta^{(k)}
= \begin{bmatrix}
\delta_{1}^{(k)} \\[6pt]
\delta_{2}^{(k)} \\[6pt]
\vdots \\[6pt]
\delta_{N}^{(k)} \\[6pt]
\end{bmatrix}
, \quad
\delta^{(k+1)}
= \begin{bmatrix}
\delta_{1}^{(k+1)} \\[6pt]
\delta_{2}^{(k+1)} \\[6pt]
\vdots \\[6pt]
\delta_{M}^{(k+1)} \\[6pt]
\end{bmatrix}
$$
and
$$
\sigma'(z^{(k)})_{N\times1}
= \begin{bmatrix}
\sigma'(z_{1}^{(k)}) \\[6pt]
\sigma'(z_{2}^{(k)}) \\[6pt]
\vdots \\[6pt]
\sigma'(z_{N}^{(k)}) \\[6pt]
\end{bmatrix}
$$

Hence, the `activation gradient` can be denoted as
$$
\delta^{(k)} 
= \left( (W^{(k+1)})^T \ \delta^{(k+1)} \right)
\odot \sigma'(z^{(k)}) 
$$

`Loss Gradient`
Using the above 2 vectors, we can denote the `loss gradient` $w.r.t$ weights in layer $k$ as
$$
\frac{\partial L}{\partial W^{(k)}}
= \delta^{(k)} \left( a^{(k-1)} \right)^T
$$
and `loss gradient` $w.r.t$ bias in layer $k$ as
$$
\frac{\partial L}{\partial b^{(k)}}
= \delta^{(k)}
$$

---

`Batch`
In real-world computation, these gradients are computed over batches
Let $B$ be the no. of samples in a batch.

Then, the `activation gradient` across all samples in batch is vectorized as
$$
\Delta^{(k)} 
= \begin{bmatrix}
| & | & & | \\[6pt]
\delta^{(k, \ 1)} & \delta^{(k, \ 2)} & \dots & \delta^{(k, \ B)}\\[6pt]
| & | & & | \\[6pt]
\end{bmatrix}
$$
and the `hidden state` from previous layer is vectorized as
$$
A^{(k-1)}
= \begin{bmatrix}
| & | & & | \\[6pt]
a^{(k-1, \ 1)} & a^{(k-1, \ 2)} & \dots & a^{(k-1, \ B)}  
\\[6pt]
| & | & & | \\[6pt]
\end{bmatrix}
$$

`Weight Gradient`
$$
\frac{\partial L}{\partial W^{(k)}}
= \frac{1}{B} \ \Delta^{(k)} \left( A^{(k-1)} \right)^T
$$

`Bias Gradient`
$$
\frac{\partial L}{\partial b^{(k)}}
= \frac{1}{B} \Delta^{(k)} \cdot \mathbf{1}
$$

---
## Algorithm
`Forward Pass`
1. Randomly initialize the weights to be close to zero
2. Feed $x$ into the [[Feed-Forward Neural Network]] input layer and compute the outputs of `input neurons`
3. Propagate the outputs of each `hidden layer` forward, in order to compute the outputs of all hidden neurons
4. Compute the final `output neuron(s)`
5. Compute the [[Loss Function]]

`Backward Pass`
1. Compute `loss gradient` of the final outputs: $\delta^{(L)} = \frac{\partial L}{\partial z^{(L)}}$
   - `MSE + Sigmoid`: $\delta^{(L)} = \left( a^{(L)} - y \right) \odot \sigma'(z^{(L)})$
   - `Cross-Entropy + Softmax`: $\delta^{(L)}  = a^{(L)} - y$
2. For each hidden layer $k$, compute the `activation gradient`:
   $\delta^{(k)} = \left( (W^{(k+1)})^T \ \delta^{(k+1)} \right) \odot \sigma'(z^{(k)})$
3. Compute the `loss gradients` of current layer $k$: 
   $\frac{\partial L}{\partial W^{(k)}}$ and $\frac{\partial L}{\partial b^{(k)}}$
4. Update the `weights` and `bias` according to the [[Gradient Descent]]
   - $W^{(k)} \leftarrow W^{(k)} - \eta \ \frac{\partial L}{\partial W^{(k)}}$
   - $b^{(k)} \leftarrow b^{(k)} - \eta \ \frac{\partial L}{\partial b^{(k)}}$
   
---
## See Also
- [[Neural Network]]
- [[Gradient Descent]]
- [[Loss Function]]