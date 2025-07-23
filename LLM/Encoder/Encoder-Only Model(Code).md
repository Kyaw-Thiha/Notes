#ml #llm/transformers/encoders 
## Minimal Encoder (without library)
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
    pos = np.arange(seq_len)[:, None]
    i = np.arange(d_model)[None, :]
    angle = pos / np.power(10000, (2 * (i//2)) / d_model)
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    pe[:, 1::2] = np.cos(angle[:, 1::2])
    return pe

def attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)
    weights = softmax(scores)
    return np.matmul(weights, V)

class EncoderLayer:
    def __init__(self, d_model, d_ff):
        self.Wq = np.random.randn(d_model, d_model)
        self.Wk = np.random.randn(d_model, d_model)
        self.Wv = np.random.randn(d_model, d_model)
        self.W1 = np.random.randn(d_model, d_ff)
        self.W2 = np.random.randn(d_ff, d_model)

    def forward(self, x):
        Q = x @ self.Wq
        K = x @ self.Wk
        V = x @ self.Wv
        attn = attention(Q, K, V)
        x = layer_norm(x + attn)
        ff = np.maximum(0, x @ self.W1)
        ff = ff @ self.W2
        x = layer_norm(x + ff)
        return x

def encoder(x, num_layers=2, d_model=64, d_ff=128):
    seq_len = x.shape[1]
    x += positional_encoding(seq_len, d_model)
    for _ in range(num_layers):
        layer = EncoderLayer(d_model, d_ff)
        x = layer.forward(x)
    return x

# Example input
x = np.random.randn(1, 6, 64)  # batch=1, seq_len=6, d_model=64
out = encoder(x)
print("Output shape:", out.shape)
```

---
## Pytorch Encoder
```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * -math.log(10000.0) / d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class EncoderOnlyBlock(nn.Module):
    def __init__(self, d_model=768, nhead=12, d_ff=3072, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class EncoderOnlyModel(nn.Module):
    def __init__(self, vocab_size=30522, d_model=768, nlayer=12, nhead=12, d_ff=3072, max_len=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList([
            EncoderOnlyBlock(d_model, nhead, d_ff) for _ in range(nlayer)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.embed(x)
        x = self.pos_enc(x)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

# Example
model = EncoderOnlyModel()
x = torch.randint(0, 30522, (2, 32))  # batch=2, seq_len=32
out = model(x)
print(out.shape)  # (2, 32, 768)
```

## BERT Real-World Model (with Tokenizer & MLM)
```python
from transformers import BertTokenizer, BertForMaskedLM
import torch

# Load pretrained BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

# Example text with a [MASK]
text = "The quick brown [MASK] jumps over the lazy dog"
inputs = tokenizer(text, return_tensors="pt")
mask_index = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

# Run model
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Decode prediction at [MASK]
predicted_token_id = logits[0, mask_index, :].argmax(dim=-1)
predicted_word = tokenizer.decode(predicted_token_id)
print(f"Prediction for [MASK]: {predicted_word}")
```

