# Vectorizing Neural Network
#ml/models/neural-network/vectorizing 

`Problem Setting`
Consider the following neural network.

![[Neural Network.png|500]]

---
`Input Layer`
Akin to how we vectorize augmented input matrix in [[Multi-Dim Input Linear Regression]],
$$
\tilde{x} = \begin{bmatrix}
1 \\
x_{1} \\
x_{2}
\end{bmatrix}
$$

`Layer-1`
We can define the `augmented weight matrix` as 
$$
\tilde{W}^{(1)} 
= \begin{bmatrix}
b_{1}^{(1)} & w_{11}^{(1)} & w_{12}^{(1)}  \\[6pt]
b_{2}^{(1)} & w_{21}^{(1)} & w_{22}^{(1)}  \\
\end{bmatrix}
$$
Hence, the outputs (`hidden states`) can be defined as
$$
\sigma \left( \tilde{W}^{(1)} \ \tilde{x} \right)
= \begin{bmatrix}
\sigma(z_{1}^{(1)}) \\[6pt]
\sigma(z_{2}^{(1)}) \\[6pt]
\end{bmatrix}
= \begin{bmatrix}
a_{1}^{(1)} \\[6pt]
a_{2}^{(2)} \\
\end{bmatrix}
= a^{(1)}
$$

`Layer-2`
We can define the `augmented weight matrix` as 
$$
\tilde{W}^{(1)} 
= \begin{bmatrix}
b_{1}^{(2)} & w_{11}^{(2)} & w_{12}^{(2)}  \\[6pt]
\end{bmatrix}
$$
Likewise, we can augment the `previous hidden state` as
$$
\tilde{a}^{(1)} = \begin{bmatrix}
1 \\
a_{1}^{(1)} \\
a_{2}^{(1)}
\end{bmatrix}
$$
Hence, the outputs (`hidden states`) can be defined as
$$
\sigma \left( \tilde{W}^{(2)} \ \tilde{a}^{(1)} \right)
= \begin{bmatrix}
\sigma(z_{1}^{(2)}) \\[6pt]
\end{bmatrix}
= \begin{bmatrix}
a_{1}^{(2)} \\[6pt]
\end{bmatrix}
= a^{(2)}
$$

---
`Alternative Problem Setting`

It can be helpful to denote `pre-activation state` for better understanding [[Backpropagation]].

![[Neural-Network-2.png|500]]

`Input layer` and `hidden states` will be vectorized as above.

`Layer-1`
We can get the `hidden state` as
$$
\sigma(\tilde{W} \ \tilde{x})
= \begin{bmatrix}
\sigma(z_{1}^{(1)}) \\[6pt]
\sigma(z_{2}^{(2)})
\end{bmatrix}
= \begin{bmatrix}
a_{1}^{(1)} \\[6pt]
a_{2}^{(1)} \\[6pt]
\end{bmatrix}
$$

This means we can denote the `augmented hidden state` as
$$
\tilde{a} 
= \tilde{\sigma}(z)
= \begin{bmatrix}
1 \\[6pt]
\sigma(z_{1}) \\[6pt]
\vdots \\[6pt]
\sigma(z_{M}) \\[6pt]
\end{bmatrix}
= \begin{bmatrix}
1 \\[6pt]
\sigma(\vec{z})
\end{bmatrix}
$$

`Output Layer`
We can denote the outputs from the output layer as
$$
\begin{align}
\hat{y} 
&= a^{(2)} \\[6pt]

&= \begin{bmatrix}
a_{1}^{(2)}
\end{bmatrix} \\[6pt]

&= \sigma(z^{(2)}) \\[6pt]

&= \sigma(\tilde{W}^{(2)} \ \tilde{a}^{(1)}) \\[6pt]

&= \sigma(\tilde{W}^{(2)} \ \tilde{\sigma}(\tilde{W}^{(1)} \ \tilde{a}^{(0)}) ) \\[6pt]

&= \sigma(\underbrace{\tilde{W}^{(2)} \ \tilde{\sigma}(z^{(1)})}_{ z^{(2)}} ) \\[6pt]
\end{align}
$$

---
## See Also
- [[Neural Network]]
- [[Backpropagation]]