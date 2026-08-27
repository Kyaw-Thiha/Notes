# Batch Normalization
We can normalize [[Activation Function|activations]] batch-wise for each [[Neural Network|layer]].
![|300](https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2F68cddd98ed9529e2b0edac143a47ec1b5ecbadd3-800x521.png&w=1920&q=75)

The following is an example of $3$ input samples and $4$ features.
![|300](https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2F409b7645d3bdc19d267f6a6bea3bbf75f70636f7-800x535.png&w=3840&q=75)

To allow the [[Neural Network|network]] to learn fluctuating distribution, we introduce
$$
y_{i} = \gamma \hat{x}_{i} + \beta
$$
where 
- $\gamma$ is the learnable scaling factor for the layer
- $\beta$ is the learnable offset for the layer 

Hence, there is a mean and standard deviation computed for each layer per [[Batch Size|batch]]. And there is a learnable scaling factor $\gamma$ and offset $\beta$ for each layer across all batches.

---
## Inference
Given that we use mean and variance per batch, we need a mean and variance for actual inference too.

For this, we maintain a moving average per [[Batch Normalization|BatchNorm layer]].
$$
\begin{align}
\hat{\mu}_{i} &= \alpha \hat{\mu}_{i} + (1 - \alpha) 
\mu_{i} \\[6pt]
\hat{\sigma}_{i} &= \alpha\hat{\sigma}_{i} + (1 - \alpha) \sigma_{i}
\end{align}
$$
where
- $\hat{\mu}_{i}$ is the moving mean for layer $i$
- $\hat{\sigma}_{i}$ is the moving variance for layer $i$
- $\alpha$ is usually set close to $1$

---
## Algorithm
Let $B = \{ x_{1}, \dots, x_{m} \}$ be the batch of data.
Let $\gamma$ and $\beta$ be the parameters to learn.
Let $\{ y_{i} = \text{BN}_{\gamma,\beta}(x_{i}) \}$ be the outputs.
Then,
$$
\begin{align}
&\mu_{\beta} \leftarrow \frac{1}{m} \sum^{m}_{i=1} 
x_{i}  & \text{Mini-Batch Mean} \\[6pt]

&\sigma^{2}_{\beta} \leftarrow \frac{1}{m} \sum^{m}_{i=1}
(x_{i} - \mu_{\beta})^{2} & \text{Mini-Batch Variance} \\[6pt]

&\hat{x}_{i} \leftarrow \frac{x_{i} - \mu_{\beta}} 
{\sqrt{ \sigma^{2}_{\beta} + \epsilon }}
&\text{Normalize} \\[6pt]

&y_{i}  \leftarrow \gamma \hat{x}_{i} + \beta 
&\text{Scale and Shift Parameters}
\end{align}
$$

---
## Pros and Cons
The benefits of using [[Batch Normalization|batch normalization]] involves
- Higher [[learning rate]]
- Acts as a [[Regularization|regularization]] for the model
- Less sensitivity to initialization

The drawbacks involves
- Does not work with [[Gradient Descent|stochastic gradient descent]]
- In small batch size, sample mean and variance are not representative of the actual distribution
- Not suited for sequence models. They could have different lengths and smaller batch sizes corresponding to longer sequences.

---
