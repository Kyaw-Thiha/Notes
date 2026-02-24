## Encoder 
The **encoder** is made of
1. Condition embedding MLP for class condition.
   (`Linear` $\to$ `GELU` $\to$ `Linear`)
2. Linear projection of spectrum tokens.
   ($1$ $\to$ $\text{model}_{\text{dim}}$ per wavelength token)
3. Encoder Transformer Block $\times 6$
4. Mean Pooling over sequence
5. Project to Gaussian distribution with
   - Projection for mean 
   - Projection for log-variance

Each **Encoder Transformer Block** has
1. RMSNorm
2. Multi-Head Attention (with 8 heads)
3. Residual Connection
4. RMSNorm
5. Cross-Attention with condition embedding
6. Residual Connection
7. RMSNorm
8. Feed-Forward Network (GELU + 4x expansion)
9. Residual Connection

### Decoder
It has two paths: one for class condition, and another on latent.

The decoder is made of
1. Global path on condition embedding
   `Linear` $\to$ `GELU` $\to$ `Dropout` $\to$ `Linear` to output spectrum
2. Local path latent projection (`linear`) + repeat 
   ($\text{latent}_{dim} \to \text{model}_{dim}$, then repeat across sequence)
3. Add sinusoidal positional encoding to local tokens.
4. Decoder Transformer block $\times 2$
5. Learned projection (`linear`) for 
   $\text{model}_{dim} \to \text{output}_{dim}$ on local path
6. Fuse global spectrum and local spectrum with learnable local-mix weight and global warmup scale.
7. Sigmoid activation function

The **Decoder Transformer Block** is made of
1. RMSNorm
2. Multi-Head Attention (with 8 heads)
3. Residual Connection
4. Gated FiLM to inject condition
5. RMSNorm
6. Feed-Forward Network (`GELU` + $4\times$ expansion)
7. Residual Connection

The **Gated FiLM** is made of
1. Gamma linear layer
2. Beta linear layer
3. A learnable scalar gate parameter 
   - initialized from a gate probability (via logit), 
   - then applied through sigmoid during forward.

---
### Hyperparams
### Main Hyperparams (Best Recipe: run_K_lat12)

  - latent_dim: 12
  - d_model: 128
  - n_heads: 8
  - encoder_layers: 6
  - decoder_layers: 2
  - dropout: 0.0
  - condition_dropout: 0.35
  - decoder_use_film: true
  - gated_film_init: 0.20
  - latent_fuse_weight (init): 0.85
  - latent_fuse_weight_min: 0.75
  - latent_fuse_weight_learnable: true
  - global_path_hidden_dim: 32
  - global_path_dropout: 0.50
  - global_path_warmup_hold_epochs: 20
  - global_path_warmup_ramp_epochs: 30
  - decoder_logit_gain: 1.0
  - loss: beta_vae
  - recon: mse
  - beta: 0.02
  - free_bits_total: 2.0
  - grad_weight: phase-1 0.0, phase-2 1.0
  - grad_metric: mse
  - grad_diff_orders: [1, 2]
  - grad_order_weights: [1.0, 0.1]
  - kl_anneal: true (linear, start 0.0, warmup ratio 0.5)
  - masked_wavelength_ranges_nm: [1700, 1900]
  - masked_wavelength_weight: 0.1
  - optimizer: Adam, lr=2e-4, weight_decay=0.0
  - scheduler: cosine (T_max=100)
  - batch_size: 32
  - max_epochs: phase-1 40, phase-2 80 (resume from phase-1)
  - seed: 42
  - input_dim: 210 (400–2490 nm, step 10)
  - condition_dim: 3
  - split: 0.8 / 0.1 / 0.1

**Dimensions to Try**
$d_{\text{model}}$: $256, 512, 768, 1024$
$d_{\text{latent}}$: $32, 64, 128, 256$

[Conditioning Mechanism Redesign Paper](https://openreview.net/forum?id=rJlHea4Kvr)
[Dual Path Paper](https://www.mdpi.com/1099-4300/27/4/423)