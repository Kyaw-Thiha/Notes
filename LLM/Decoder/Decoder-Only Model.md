# 🤖 Decoder-Only Models in Transformers

Decoder-only models are a streamlined version of the original Transformer architecture. They **discard the encoder stack** and focus purely on generating sequences in an **autoregressive** fashion.

These models form the backbone of powerful generative models like **GPT-2**, **GPT-3**, **LLaMA**, and **Mistral**.

---
## 📌 Key Concept

A decoder-only model processes input tokens **one directionally** — from **left to right** — and is trained to **predict the next token** given all the previous ones.

At training time, for a sequence $x = [x_1, x_2, \dots, x_T]$, the objective is:

$$
\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t \mid x_1, \dots, x_{t-1})
$$

---
## 🧱 Architecture Overview

Each decoder block consists of:

1. **Masked Multi-Head Self-Attention**
   - Allows each token to attend to **previous tokens only**
   - Uses a **causal mask** to enforce this

2. **Feed-Forward Neural Network (FFN)**
   - Applies a two-layer MLP with ReLU or GELU to each token independently

3. **Add & Norm Layers**
   - Residual connections followed by layer normalization

4. **Positional Encoding**
   - Injects token position since attention is permutation-invariant

---
## 🔄 Transformer Decoder Layer Flow

```text
Input Tokens
    ↓
Token Embedding + Positional Encoding
    ↓
────────────────────────────
Repeat N times:
    ↓ Masked Self-Attention
    ↓ Add & Norm
    ↓ Feed-Forward Network
    ↓ Add & Norm
────────────────────────────
    ↓
Final Linear Layer → Vocabulary logits
    ↓
Softmax → Next token probability
```

---
## 🧠 Causal Masking

To prevent a token from “seeing the future,” we use a **causal (triangular) mask**:

For a sequence of 4 tokens, the attention mask looks like:

$$
\begin{bmatrix}
1 & 0 & 0 & 0 \\\\
1 & 1 & 0 & 0 \\\\
1 & 1 & 1 & 0 \\\\
1 & 1 & 1 & 1
\end{bmatrix}
$$

This ensures that the $t$-th token can only attend to tokens $0 \dots t$.

---
## 🔁 Autoregressive Inference

At inference time, decoder-only models **generate tokens one by one**:

```text
Prompt: "The cat"
    ↓
Predict: "sat"
    ↓
Prompt: "The cat sat"
    ↓
Predict: "on"
    ↓
Prompt: "The cat sat on"
    ↓
Predict: "the"
...
```

This loop continues until a stopping condition is reached (like an `<EOS>` token or max length).

---

## 🔬 Comparison to Encoder-Decoder Models

| Feature               | Decoder-Only            | Encoder-Decoder           |
|----------------------|-------------------------|---------------------------|
| Architecture         | Only decoders           | Encoders + Decoders       |
| Input                | Partial sequence (causal) | Source + target sequences |
| Use Case             | Language modeling, generation | Translation, summarization |
| Example Models       | GPT, LLaMA, Mistral      | T5, BART, MarianMT         |

---
## 🔧 Applications

Decoder-only models are used in:

- 📝 Text generation (e.g., GPT)
- 💬 Chatbots
- ✍️ Code completion (e.g., Codex)
- 📚 Language modeling (e.g., LLaMA)
- 🔄 Few-shot learning via in-context prompts

---
## ✅ Summary

- Trained to predict the next token from previous ones
- Only includes Transformer decoder blocks
- Uses causal masking for autoregressive generation
- Efficient, scalable, and highly parallelizable during training
- The foundation for many modern LLMs

---
## 🧪 PyTorch Pseudocode (Minimal Block)
```python
class DecoderOnlyBlock(nn.Module):
    def __init__(self, d_model, nhead, d_ff):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x2, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + x2)
        x2 = self.ff(x)
        x = self.norm2(x + x2)
        return x
```

For more code examples,
- [[Decoder-Only Model(Code)#Minimal Decoder-Only Model (no library)|Minimal Example]]
- [[Decoder-Only Model(Code)#Pytorch Decoder-Only Model|Pytorch Example]]
- [[Decoder-Only Model(Code)#GPT-2 Style Custom Implementation|Custom GPT-2 Example]]

---
## See Also
- [[Decoder]]
- [[Encoder-Only Model]]
- [[Self-Attention]]
