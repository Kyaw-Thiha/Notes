#ml #llm/transformers/encoders 
## Minimal Encoder (with No Libraries)
```python
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def layer_norm(x, eps=1e-6):
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + eps)

def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates

    # Apply sin to even indices and cos to odd indices
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angle_rads[:, 0::2])
    pe[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return pe

def encoder_layer(X, W_q, W_k, W_v, W1, b1, W2, b2):
    # QKV projection
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v

    # Scaled dot-product attention
    dk = Q.shape[-1]
    attention_scores = Q @ K.T / np.sqrt(dk)
    attention_weights = softmax(attention_scores)
    attention_output = attention_weights @ V

    # Add & Norm (post-attention)
    X2 = layer_norm(X + attention_output)

    # Feedforward Network
    hidden = np.maximum(0, X2 @ W1 + b1)  # ReLU
    ff_output = hidden @ W2 + b2

    # Add & Norm (post-FFN)
    output = layer_norm(X2 + ff_output)

    return output

def stack_encoder(X, num_layers, d_model, d_ff):
    np.random.seed(0)
    seq_len = X.shape[0]

    # Add positional encoding
    pe = positional_encoding(seq_len, d_model)
    X = X + pe

    for _ in range(num_layers):
        # Random weights per layer (for simplicity)
        W_q = np.random.randn(d_model, d_model)
        W_k = np.random.randn(d_model, d_model)
        W_v = np.random.randn(d_model, d_model)
        W1 = np.random.randn(d_model, d_ff)
        b1 = np.random.randn(d_ff)
        W2 = np.random.randn(d_ff, d_model)
        b2 = np.random.randn(d_model)

        X = encoder_layer(X, W_q, W_k, W_v, W1, b1, W2, b2)

    return X

# Sample 3 tokens with d_model = 4
X = np.array([
    [1.0, 0.5, 0.2, 0.1],
    [0.0, 1.0, 0.5, 0.0],
    [0.3, 0.3, 0.3, 0.3]
])

output = stack_encoder(X, num_layers=2, d_model=4, d_ff=8)
print("Final Output:\n", output)

```

---
## Real World Encoder (Pytorch)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # Self-attention + residual
        attn_output, _ = self.self_attn(x, x, x, attn_mask=src_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # FFN + residual
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, d_model=128, nhead=8, d_ff=512, num_layers=6, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask)
        return x

# Example usage
batch_size = 2
seq_len = 10
d_model = 128

x = torch.randn(batch_size, seq_len, d_model)  # [B, T, D]
encoder = TransformerEncoder(d_model=d_model, nhead=8, d_ff=512, num_layers=6)
output = encoder(x)
print("Final encoder output shape:", output.shape)

```

## See Also
- [[Encoder]]
- [[Encoder-Only Model]]



