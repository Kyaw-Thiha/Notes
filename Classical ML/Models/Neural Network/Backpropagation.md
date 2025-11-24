# Backpropagation
#ml/models/neural-network/backpropagation 

`Backpropagation` is the main algorithm that enables the [[Neural Network]] to learn.

![Backpropagation](https://miro.medium.com/v2/resize:fit:1280/0*tSkOl1glLC8K79D-.gif)

---
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
## Algorithm
`Forward Pass`
1. Randomly initialize the weights to be close to zero
2. Feed $x$ into the [[Feed-Forward Neural Network]] input layer and compute the outputs of `input neurons`
3. Propagate the outputs of each `hidden layer` forward, in order to compute the outputs of all hidden neurons
4. Compute the final `output neuron(s)`
5. Compute the [[Loss Function]]

`Backward Pass`
1. Compute `loss gradient` of the final outputs: $\delta^{(L)} = \frac{\partial L}{\partial z^{(L)}}$
2. Recursively compute the `loss gradient` of each hidden layer:
   $\delta^{(k)} = \sigma^{'}(z_{i}^{(k)}) \sum^m_{j=1} \delta_{j}^{(k+1)} w_{j,i}^{(k+1)}$
3. Compute the `loss w.r.t weights`: $\frac{\partial L}{\partial \tilde{W}^{(k-1)}}$
4. Update the `weights` according to the [[Gradient Descent]]