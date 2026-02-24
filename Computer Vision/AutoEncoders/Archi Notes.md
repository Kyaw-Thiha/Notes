## Summary
The encoder has $6$ transformer blocks and decoder has $4$ transformer blocks.

Each transformer block has
1. RMSNorm
2. Multi-Head Attention (with 8 heads)
3. Dropout
4. Residual Connection
5. RMSNorm
6. Feed-Forward Network (GELU + 4x expansion)
7. Dropout $(0.1)$
8. Residual Connection

In order to project from encoder to latent, we will use `CLS token`
In order to project from latent to decoder, we will use `repeated with positional encoding`
In order to learn local features, we will be adding a `local_refiner block`.


For condition injection, we will be using FiLM layers (2 linear layer) and injecting in every layer.

For positional encoding, we will be using wavelength-aware encoding (every $10m$).

In order to prevent posterior collapse, we will be using `Scale-VAE`.

$d_{\text{model}}$: $256, 512, 768, 1024$
$d_{\text{latent}}$: $32, 64, 128, 256$

---
## Local Refiner Block
Local refiner block has

  1. Depthwise Conv1d $(k=7, p=3)$ for broad local spectral context $(~70nm \text{ window})$
  2. GELU activation
  3. Pointwise Conv1d $(1 \times 1)$ expanding channels $(d_{model} \to 2 \times d_{model})$
  4. GELU activation
  5. Pointwise Conv1d $(1 \times 1)$ projecting back $(2 \times d_{model} \to d_{model})$
  6. Depthwise Conv1d $(k=3, p=1)$ for fine local detail $~30nm \text{ window}$
  7. GELU activation
  8. Pointwise Conv1d $(1 \times 1)$ for final channel mixing
  9. Residual add with transformer output 

---

## Architecture

- Number of heads: 8 is standard, 4-16 range
- Pre-norm (normalize before attention)
- RMS Norm
- FFN: GeLU + 4x expansion
- Depth: 6 in encoder and 4 in decoder
- Dropout: 0.1

**One Transformer Layer**
1. RMSNorm
2. Multi-Head Attention (with 8 heads)
3. Dropout
4. Residual Connection
5. RMSNorm
6. Feed-Forward Network (GELU + 4x expansion)
7. Dropout
8. Residual Connection


**Fixed Latent Dimension** (Encoder -> Latent)
- CLS Token
- Q-Former
- Max Pooling

**Latent to Sequence** (Latent -> Decoder)
- Repeat with Positional Encoding
- Learned decoder tokens
- Cross-attention decoder

**Condition Injection**
- Adaptive LayerNorm (AdaLN)

**Positional Encoding**
- Custom wavelength-aware encoding
- Or RoPE (Rotary Position Embeddings)

---
The Solution:
Penalize incorrect slopes (rate of change) between adjacent wavelengths.

Implementation:
- Compute spectral gradients: `torch.diff(spectrum, dim=1)`
- Compare generated vs target gradients using MSE (or L1)
- Add weighted gradient loss to reconstruction:

 $$L_{\text{recon}} = L_{\text{MSE}} + \lambda \cdot L_{\text{grad}}$$

Total VAE loss: 
$$L_{total}=L_{\text{recon}} + L_{\text{KL}}$$

Key Parameters:
- $\lambda$ (grad_weight): $(0.5 - 1.0)$ recommended for spectral data
- $\text{diff}_{\text{order}} = 1$: Match sharpness $(\text{slopes})$
- $\text{diff}_{\text{order}} = 2$: Match curvature $(\text{peak shapes})$

---
## Preventing Posterior Collapse
- Scale-VAE
- $\beta-\text{vae}$ 
- KL-annealing (Cyclic?)
- Number of layers in decoder is less than number of layers in encoder
- If not working, use non-transformer based decoder

---
## Encouraging non-smoothness
- Add `gradient loss`
  Encourage model to match the steepness of data
- Add `1D CNN` 
  The local refinder block
- Feature aware loss
  Higher weighted loss at specific spectra band