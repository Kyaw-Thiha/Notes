# AutoEncoders

**Auto-Encoders** a specific types of models which 
- `Compression`: encoder maps $x \to \text{latent } z$ (lower dim)
- `Reconstruction`: decoder maps $z \to \text{reconstructed } \hat{x}$ 

The training attempts to make $\hat{x}$ as close as possible to $x$.

![Auto-Encoders](https://framerusercontent.com/images/0nBS6F387hiWsQcKmx1k81iN8U.png)

Since `latent space` is smaller than input space, the model is forced to learn the most important features.

This makes `autoencoders` useful for:
- **Dimensionality Reduction**
- **Feature Learning**
- **Denoising**

> IMPORTANT: The `encoder` and `decoder` used here are not the same to the ones from the `Transformer` models.

## Variants
- `Basic AutoEncoder`
  Deterministic mapping $x \to z \to \hat{x}$
- `Denoising AutoEncoder`
  Input is corrupted, target is original
- `Sparse AutoEncoder`
  Force most activations in $z$ to be near-zero (encourage compact codes)
- `Convolutional AutoEncoder`
  Use [[Convolution Layer]] for images
- `Variational AutoEncoder`
  Encoder outputs a distribution

