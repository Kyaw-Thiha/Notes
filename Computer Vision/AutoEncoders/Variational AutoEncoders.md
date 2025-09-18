# Variational AutoEncoders (VAE)
Compared to `Basic AutoEncoders`, instead of producing one latent vector per input, the `encoder` outputs 2 things:
- Mean: $\mu(x)$
- Log-Variance: $\log(\alpha^2 (x))$

![Variational AutoEncoder Architecture](https://media.geeksforgeeks.org/wp-content/uploads/20231201153426/Variational-AutoEncoder.png)

They can be used to define a `Gaussian Distribution`: 
$$
q(z | x) = N(z; \mu(x), \alpha^2(x).I)
$$

---
## Why Log-Variance instead of Variance?
The encoder outputs a log variance and then compute $\alpha^2 = e^{\log(\alpha^2)}$
This guarantees
- $\alpha^2 > 0$
- Numerical Stability: Prevent underflow/overflow from values blowing up
- Training Convenience: `KL Divergence` has terms like $\log(\alpha^2)$

---
## Reparameterization Trick

![Variational AutoEncoder Architecture](https://miro.medium.com/v2/1*r1R0cxCnErWgE0P4Q-hI0Q.jpeg)

Recall that encoder defines a distribution $q(z | x) = N(\mu(x), \alpha^2(x).I)$.

When computing gradient loss, if we sample $z \sim N(\mu, \alpha^2)$ directly, the **randomness** will block gradients.
We cannot back-propagate through a 'random draw'.

Instead, we first draw $\epsilon \sim N(0, 1)$ 
Then, we scale it by variance and shift it by mean:   $z = \mu + \sigma.\epsilon$ .

One way to see it is that we are teaching the model to update its $\mu$ and $\log(\alpha^2)$ to fit for different values of $\epsilon$ from the standard normal distribution.

---
## Decoder
**Input**: A latent sample $z$ (from `parameterization` trick)
**Output**: Probability distribution of data $x$

For images, each pixel output of decoder could be treated as
- Output of `Bernoulli Distribution` if modelled as discrete
- mean of `Gaussian Distribution` if modelled as continuous

This mean that there are 2 sources of randomness - `Latent Gaussian` and `Output Bernoulli/Gaussian`.
- `Latent Gaussian` controls **global variations**
- `Output Bernoulli/Gaussian` sampling controls **local variations**.

---
## Loss
Make reconstructions accurate and keep latent space smooth.  

$$
\begin{align}

\mathcal{L}(x) &= \mathcal{L}_{rec}(x) + D_{KL}(q(z \mid x) \,\|\, p(z))

\\[10pt]

&= \mathbb{E}_{q_\phi(z \mid x)}[-\log p_\theta(x \mid z)] + D_{KL}(q_\phi(z \mid x) \,\|\, p(z))
\end{align}
$$
**Reconstruction Loss**: Make output close to input
**KL Term**: Keep latent distribution close to standard normal

### `Reconstruction Loss`  
Decoder outputs pixel probabilities $\hat{x}_i$.  
For binary MNIST:  

$$
\mathcal{L}_{rec}(x) = - \sum_i \big[ x_i \log \hat{x}_i + (1-x_i)\log(1-\hat{x}_i) \big]
$$

### `KL Divergence`  
Regularizes latent posterior toward prior $p(z) = \mathcal{N}(0, I)$.  

$$
D_{KL}(q(z \mid x) \,\|\, p(z)) = -\tfrac{1}{2} \sum_j \Big( 1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2 \Big)
$$

---
## See Also
- [[AutoEncoders]]
- [[Conditional Variational AutoEncoders]]
