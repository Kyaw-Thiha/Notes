#ml #llm/transformers/attention-mask 
# Attention Mask Mechanism

The attention mask is a matrix that **controls which tokens each token can attend to** in the self-attention process.

---
## Purpose

Masks are used to:
- Prevent attending to **padding tokens**.
- Enforce **causal/temporal constraints** (e.g., GPT models should not "see the future").

---
## Mask Types

### 1. **Padding Mask**
Used to prevent attention to padded positions in a batch.

For sequence `[A, B, PAD]`, the mask might look like:

$$
\begin{bmatrix}
1 & 1 & 0 \\
1 & 1 & 0 \\
1 & 1 & 0 \\
\end{bmatrix}
$$

`1` = attend, `0` = mask (ignore).

### 2. **Causal Mask (Lower Triangular)**
Used in **decoder-only models like GPT** to prevent a token from attending to future tokens.

For a sequence of 3 tokens:

$$
\text{Mask} =
\begin{bmatrix}
1 & 0 & 0 \\
1 & 1 & 0 \\
1 & 1 & 1 \\
\end{bmatrix}
$$

This ensures:
- Token 1 attends only to itself.
- Token 2 attends to tokens 1 and 2.
- Token 3 attends to all preceding tokens.

---
## Applied in Self-Attention

The attention scores $QK^T$ are **masked** by setting the attention to $-\infty$ (or a very large negative number) where the mask is `0`. This effectively **zeroes out these positions after softmax**.

$$
\text{masked\_scores} = QK^T + \text{mask}
$$

---
## Summary

| Model | Mask Type                            | Effect                  |
| ----- | ------------------------------------ | ----------------------- |
| BERT  | None / Padding                       | Full bidirectional      |
| GPT   | Causal mask                          | Left-to-right only      |
| T5    | Causal (decoder) + Padding (encoder) | Both restrictions apply |

---
## See Also
- [[Self-Attention]]
- [Vaswani et al., "Attention is All You Need" (2017)](https://arxiv.org/pdf/1706.03762)
