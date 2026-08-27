# AutoEncoders

**Auto-Encoders** a specific types of models which 
- `Compression`: encoder maps $x \to \text{latent } z$ (lower dim)
- `Reconstruction`: decoder maps $z \to \text{reconstructed } \hat{x}$ 

The training attempts to make $\hat{x}$ as close as possible to $x$.

![Auto-Encoders|300](https://framerusercontent.com/images/0nBS6F387hiWsQcKmx1k81iN8U.png)

Since [[AutoEncoders|latent space]] is smaller than input space, the model is forced to learn the most important features.

This makes [[AutoEncoders]] useful for:
- **Dimensionality Reduction**
- **Feature Learning**
- **Denoising**

> IMPORTANT: The `encoder` and `decoder` used here are not the same to the ones from the `Transformer` models.

---
## Variants
- [[AutoEncoders|Basic Autoencoder]]
  Deterministic mapping $x \to z \to \hat{x}$.
- [[Convolutional CVAE]]
  Use [[Convolution Layer]] for images.
- [[Variational AutoEncoders (VAE)]]
  Encoder outputs a distribution. Smooth latent space.
- [[Conditional Variational AutoEncoders (CVAE)]]
  Allows injecting condition into decoder.

---
## See Also
- [[AutoEncoders]]
- [[Variational AutoEncoders (VAE)]]
- [[Conditional Variational AutoEncoders (CVAE)]]
- [[Convolutional CVAE]]
- [[Maths Behind VAE]]
- [[Posterior Collapse of CVAE]]
