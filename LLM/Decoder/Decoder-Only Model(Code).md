#ml #llm/transformers/decoders 
## Minimal Decoder-Only Model (no library)
```python
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def layer_norm(x, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + eps)

def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, None]
    i = np.arange(d_model)[None, :]
    angle = pos / np.power(10000, (2 * (i//2)) / d_model)
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    pe[:, 1::2] = np.cos(angle[:, 1::2])
    return pe

def causal_mask(seq_len):
    return np.tril(np.ones((seq_len, seq_len)))

def attention(Q, K, V, mask):
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)
    scores = np.where(mask == 0, -1e9, scores)
    weights = softmax(scores)
    return np.matmul(weights, V)

class DecoderOnlyBlock:
    def __init__(self, d_model, d_ff):
        self.Wq = np.random.randn(d_model, d_model)
        self.Wk = np.random.randn(d_model, d_model)
        self.Wv = np.random.randn(d_model, d_model)
        self.W1 = np.random.randn(d_model, d_ff)
        self.W2 = np.random.randn(d_ff, d_model)

    def forward(self, x, mask):
        Q = x @ self.Wq
        K = x @ self.Wk
        V = x @ self.Wv
        attn = attention(Q, K, V, mask)
        x = layer_norm(x + attn)
        ff = np.maximum(0, x @ self.W1)
        ff = ff @ self.W2
        x = layer_norm(x + ff)
        return x

def decoder_only_model(x, num_layers=2, d_model=64, d_ff=128):
    seq_len = x.shape[1]
    x += positional_encoding(seq_len, d_model)
    mask = causal_mask(seq_len)[None, :, :]
    for _ in range(num_layers):
        block = DecoderOnlyBlock(d_model, d_ff)
        x = block.forward(x, mask)
    return x

# Example input
x = np.random.randn(1, 5, 64)
out = decoder_only_model(x)
print("Output shape:", out.shape)
```

## Pytorch Decoder-Only Model
```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -math.log(10000.0) / d_model)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class DecoderOnlyModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, d_ff=2048, num_layers=6):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.blocks = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model, nhead, d_ff, batch_first=True)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)
        x = self.embed(x)
        x = self.pos_enc(x)
        for layer in self.blocks:
            x = layer(x, memory=None, tgt_mask=mask)
        x = self.norm(x)
        return self.output(x)

# Usage
model = DecoderOnlyModel(vocab_size=10000)
x = torch.randint(0, 10000, (2, 10))
logits = model(x)
print("Logits shape:", logits.shape)  # (batch, seq, vocab)
```

## GPT-2 Style Custom Implementation
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.register_buffer("mask", torch.tril(torch.ones(1024, 1024)).unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        causal_mask = self.mask[:, :seq_len, :seq_len] == 0
        x, _ = self.attn(x, x, x, attn_mask=causal_mask)
        return x

class GPTBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPTModel(nn.Module):
    def __init__(self, vocab_size, d_model=768, n_layer=12, n_head=12, d_ff=3072, max_len=1024):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, n_head, d_ff) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        b, t = x.size()
        tok = self.token_embed(x)
        pos = self.pos_embed[:, :t, :]
        x = tok + pos
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)

# Example usage
model = GPTModel(vocab_size=50257)
x = torch.randint(0, 50257, (1, 16))
out = model(x)
print(out.shape)  # (1, 16, vocab_size)
```

## See Also
- [[Decoder-Only Model]]
- [[Decoder]]