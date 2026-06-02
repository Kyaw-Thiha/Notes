# Normalization
[[Neural Network|Neural networks]] rescale input and [[Activation Function|activations]] to a standard range. This stabilizes learning, leading to higher learning rate and faster convergence.

---
## Input Normalization
We first normalize the inputs using
$$
X_{i} = \frac{X_{i} - \mu_{i}}{\sigma_{i}} 
$$
where
- $\mu_{i}$ is the mean of the [[Batch Size|batch]]
- $\sigma_{i}$ is the standard deviation of the [[Batch Size|batch]]

---
## Batch Normalization
We can then normalize [[Activation Function|activations]] batch-wise for each [[Neural Network|layer]].

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
\triangleq \text{BN}_{\gamma,\beta}(x_{i})
&\text{Scale and Shift}
\end{align}
$$

We have learnable $\gamma$ and $\beta$ to allow the [[Neural Network|network]] to learn fluctuating distribution.

We have a moving average which we use during inference time.
$$
\begin{align}
\hat{\mu}_{i} &= \alpha \hat{\mu}_{i} + (1 - \alpha) 
\mu_{i} \\[6pt]
\hat{\sigma}_{i} &= \alpha\hat{\sigma}_{i} + (1 - \alpha) \sigma_{i}
\end{align}
$$

[[Batch Normalization|Read More]]

---
## Layer Normalization
Alternatively, we could normalize [[Activation Function|activations]] across all features.

Let $B = \{ x_{11}, \dots, x_{dm} \}$ be a matrix of $m$ data with $d$ features.
Let $\gamma$ and $\beta$ be the parameters to learn.
Then,
$$
\begin{align}
&\mu_{\beta} \leftarrow \frac{1}{d} \sum^{d}_{i=1} 
x_{i}  & \text{Features Mean} \\[6pt]

&\sigma^{2}_{\beta} \leftarrow \frac{1}{d} \sum^{d}_{i=1}
(x_{i} - \mu_{\beta})^{2} & \text{Features Variance} \\[6pt]

&\hat{x}_{i} \leftarrow \frac{x_{i} - \mu_{\beta}} 
{\sqrt{ \sigma^{2}_{\beta} + \epsilon }}
&\text{Normalize} \\[6pt]

&y_{i}  \leftarrow \gamma \hat{x}_{i} + \beta 
&\text{Scale and Shift Parameters}
\end{align}
$$

[[Layer Normalization|See Also]]

---
## Differences
To summarize their differences
- [[Batch Normalization|BatchNorm]] normalize across the mini-batch while [[Layer Normalization|LayerNorm]] normalizes across all features.
- [[Batch Normalization|BatchNorm]] performs worse with small [[batch size]], while [[Layer Normalization|LayerNorm]] is independant.
- [[Batch Normalization|BatchNorm]] has different processing for training and inference, while [[Layer Normalization|LayerNorm]] uses the same.

---
## See Also
- [Good Article](https://www.pinecone.io/learn/batch-layer-normalization/)
- [[Batch Normalization]]
- [[Layer Normalization]]
- [[Neural Network]]