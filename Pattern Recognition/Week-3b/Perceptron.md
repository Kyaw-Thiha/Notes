# Perceptron
A `perceptron` is a [[Halfspace|Halfspace classifier]] as well as the algorithm to find the optimal weights for that classifier.

![Perceptron|300](https://miro.medium.com/v2/da:true/resize:fit:1200/0*Ib3_FfuOy04kOmfO)

---
## Intuition
Note that we are using 
- $0 \text{-}1$ [[Loss Function]]
- and $\mathcal{Y} \in \{ -1, +1 \}$

The `update rule` of the weights of the perceptron can be defined as

- If classification is correct,
$$
\mathbf{w}^{(t+1)}
= \mathbf{w}^{(t)} 
$$
- If classification is incorrect,
$$
\mathbf{w}^{(t+1)}
= \mathbf{w}^{(t)} + y_{i}x_{i}
$$

### Why does it work?
The idea is that we are making $\mathbf{w}$ more like the desired vector.

$$
\begin{align}
&y_{i} \langle 
\mathbf{w}^{(t+1)}, \mathbf{x}_{i} \rangle \\[6pt]
= \ &y_{i} \langle \mathbf{w}^{(t)} +  
y_{i}\mathbf{x}_{i}, \ \mathbf{x}_{i} \rangle \\[6pt]

= \ &y_{i} \langle \mathbf{w}^{(t)} +  
\mathbf{x}_{i} \rangle + ||\mathbf{x}_{i}^2||
\end{align}
$$

This makes the criterion more positive.

---
