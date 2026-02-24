# Forms of Regularization
### Large Datasets

---
### Small Models

---
### Entropy Regularization
`Entropy regularization` is used when making probabilistic predictions.
$$
\mathbf{w}^{*}
= \arg\min_{\mathbf{w} \in \mathcal{H}}
\ L_{S}(\mathbf{w}) - \lambda H(p(\mathbf{w}))
$$
where
- $H(p)$ is the `entropy` of a probability distribution $p$

The regularization term penalizes the objective function for being too certain.
(narrow distributions have less entropy)

**Alternative Entropy Regularization**
Alternatively, we might penalize the learning algorithm for making predictions that deviate too much from a specific distrubtion $G$ (usually uniform distribution).
$$
\mathbf{w}^{*} 
= \arg\min_{\mathbf{w} \in \mathcal{H}}
L_{S}(\mathbf{w}) + \lambda \ D(G \ || \ p(\mathbf{w}))
$$
where
- $D(G \ || \ p(\mathbf{w}))$ is a function measures some difference between the predicted probability distribution and the baseline distribution $G$

---
### Dropout Regularization
![Dropout Regularization|400](https://quantdare.com/wp-content/uploads/2021/05/Webp.net-gifmaker.gif)

---
### Meta Learning
`Meta-learning` uses multiple tasks to regularize the parameters of an algorithm.

![Meta-Learning|400](https://i.ytimg.com/vi/Wn40DX6Ab8Y/hq720.jpg?sqp=-oaymwE7CK4FEIIDSFryq4qpAy0IARUAAAAAGAElAADIQj0AgKJD8AEB-AH-CYAC0AWKAgwIABABGGUgZShlMA8=&rs=AOn4CLDXW8GvqKUr4UsfuAtB0P7kYAuzRQ)

At each iteration, gradients are computed $w.r.t$ a randomly selected set of tasks and used to update the model parameters.

After training, model parameters are in a position to adapt quickly to new tasks and not overfit to any one task.

---
