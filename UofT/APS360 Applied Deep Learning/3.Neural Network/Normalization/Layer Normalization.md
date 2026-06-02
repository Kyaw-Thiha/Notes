# Layer Normalization

We could normalize [[Activation Function|activations]] across all features for each [[Neural Network|layer]].

Consider the following example with $3$ batch size and $4$ features.
![|300](https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2F567b2a2d454f2da286ce3cbbe6ce4583a1e2417f-800x627.png&w=1920&q=75)

---
## Algorithm
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

---
Compared to [[Batch Normalization|batch normalization]], 
- this is simpler to implement
- and does not depend on [[batch size]]