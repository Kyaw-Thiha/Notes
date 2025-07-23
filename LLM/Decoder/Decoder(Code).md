#ml #llm/transformers/decoders 
## Minimal Decoder (without libraries)
```python
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def layer_norm(x, eps=1e-6):
    mean = x.mean(-1, keepdims=True)
    std = x.std(-1, keepdims=True)
    return (x - mean) / (std + eps)

def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return angle_rads

def attention(Q, K, V, mask=None):
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(Q.shape[-1])
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
    weights = softmax(scores)
    return np.matmul(weights, V)

class DecoderLayer:
    def __init__(self, d_model, d_ff):
        self.Wq1 = np.random.randn(d_model, d_model)
        self.Wk1 = np.random.randn(d_model, d_model)
        self.Wv1 = np.random.randn(d_model, d_model)
        
        self.Wq2 = np.random.randn(d_model, d_model)
        self.Wk2 = np.random.randn(d_model, d_model)
        self.Wv2 = np.random.randn(d_model, d_model)
        
        self.W1 = np.random.randn(d_model, d_ff)
        self.W2 = np.random.randn(d_ff, d_model)
    
    def forward(self, x, enc_out, mask):
        # Masked self-attention
        Q = np.matmul(x, self.Wq1)
        K = np.matmul(x, self.Wk1)
        V = np.matmul(x, self.Wv1)
        attn1 = attention(Q, K, V, mask)
        x = layer_norm(x + attn1)

        # Encoder-decoder attention
        Q = np.matmul(x, self.Wq2)
        K = np.matmul(enc_out, self.Wk2)
        V = np.matmul(enc_out, self.Wv2)
        attn2 = attention(Q, K, V)
        x = layer_norm(x + attn2)

        # Feedforward
        ff = np.matmul(x, self.W1)
        ff = np.maximum(0, ff)  # ReLU
        ff = np.matmul(ff, self.W2)
        x = layer_norm(x + ff)
        return x

def decoder(x, enc_out, mask, num_layers=2, d_model=64, d_ff=128):
    x = x + positional_encoding(x.shape[1], d_model)
    for _ in range(num_layers):
        layer = DecoderLayer(d_model, d_ff)
        x = layer.forward(x, enc_out, mask)
    return x

# Example usage
batch_size, seq_len, d_model = 2, 5, 64
x = np.random.randn(batch_size, seq_len, d_model)
enc_out = np.random.randn(batch_size, seq_len, d_model)
mask = np.tril(np.ones((seq_len, seq_len)))

out = decoder(x, enc_out, mask)
print(out.shape)
```

## Real-World Example (using Pytorch)
```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, tgt_mask=None):
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = self.norm1(tgt + tgt2)

        tgt2, _ = self.cross_attn(tgt, memory, memory)
        tgt = self.norm2(tgt + tgt2)

        tgt2 = self.ff(tgt)
        tgt = self.norm3(tgt + tgt2)
        return tgt

class TransformerDecoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, d_ff=2048, num_layers=6, vocab_size=10000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, d_ff) for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory, tgt_mask=None):
        x = self.embedding(tgt)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, memory, tgt_mask)
        return self.out_proj(x)

# Example usage
decoder = TransformerDecoder()
tgt = torch.randint(0, 10000, (2, 10))  # batch of token IDs
memory = torch.randn(2, 15, 512)  # encoder output
tgt_mask = nn.Transformer.generate_square_subsequent_mask(10)
out = decoder(tgt, memory, tgt_mask=tgt_mask)
print(out.shape)  # (batch_size, seq_len, vocab_size)
```

## See Also
- [[Decoder]]
