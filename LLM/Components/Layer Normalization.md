# 🧮 Mathematical Explanation of Layer Normalization

**Layer Normalization** is a technique used to stabilize and accelerate training by normalizing inputs across the **features** of each data point (token). 
Unlike **batch normalization**, which normalizes across the batch dimension, layer normalization operates **independently per sample**, making it well-suited for sequence models like Transformers.

Given an input vector $x \in \mathbb{R}^d$ (e.g., a token embedding of dimension $d$), layer norm computes:

$$
\mu = \frac{1}{d} \sum_{i=1}^{d} x_i, \quad
\sigma = \sqrt{\frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2 + \epsilon}
$$

Then each element is normalized as:

$$
\text{LayerNorm}(x_i) = \frac{x_i - \mu}{\sigma}
$$

Optionally, learnable scale and shift parameters $\gamma$ and $\beta$ are applied:

$$
\text{output}_i = \gamma_i \cdot \frac{x_i - \mu}{\sigma} + \beta_i
$$

This ensures that each token’s representation has zero mean and unit variance, which helps gradients flow more smoothly through deep layers and prevents internal covariate shift. It is typically applied **after residual connections** in the Transformer.

## See Also
- [[Encoder]]
- [[Decoder]]
``