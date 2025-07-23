#ml #llm/transformers/self-attention
# Self-Attention Mechanism

Self-attention is a core component of the Transformer architecture, enabling each token in a sequence to compute a **contextualized representation** by attending to **all other tokens**.

---
## Definition

Given an input sequence of tokens, self-attention computes a weighted representation of each token based on its relevance to other tokens in the sequence.

Mathematically, for input tokens represented as vectors, self-attention involves:

1. **Linear projections** to obtain:
   - Queries **Q**
   - Keys **K**
   - Values **V**

2. **Scaled Dot-Product Attention:**
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$
Where:
- $Q$: Query matrix
- $K$: Key matrix
- $V$: Value matrix
- $d_{k}$: Dimensionality of the key vectors

---
## Query, Key, Value Projections

Each token's input embedding $x_{i}$ (dimension $d_{\text{model}}$) is **projected** into three different spaces using learned weight matrices:
$$
Q = X W^Q
$$
$$
K = X W^K
$$
$$
V = X W^V
$$

Where:
- $X$ is the matrix of token embeddings.
- $W^Q, W^K, W^V$ are trainable parameter matrices:
  - $W^Q, W^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$
  - $W^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$

The **dot products between Query and Key** yield the **attention scores**, which measure the similarity between tokens.

---
## Scaled Dot-Product Attention
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$
### Scaling and Softmax
After computing the dot product \(QK^T\), we **scale** the result:
$$
\text{scaled\_scores} = \frac{QK^T}{\sqrt{d_k}}
$$
Then apply the softmax function **row-wise** to normalize the scores:
$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$
---
## Example

For a 3-token sequence: `["I", "like", "cats"]`
Assuming $QK^T$ yields:
$$
\begin{bmatrix}
0.9 & 0.1 & 0.0 \\
0.2 & 0.8 & 0.1 \\
0.0 & 0.3 & 0.7 \\
\end{bmatrix}
$$

After softmax scaling (row-wise):
$$
\begin{bmatrix}
0.82 & 0.15 & 0.03 \\
0.17 & 0.74 & 0.09 \\
0.05 & 0.25 & 0.70 \\
\end{bmatrix}
$$

This matrix describes **how much each token attends to the others**. For example:
- Token "I" attends mostly to itself (0.82).
- Token "cats" attends 70% to itself, 25% to "like", 5% to "I".

---
## Properties
- **Bidirectional context**: In encoder models like BERT, all tokens are visible to each other.
- **Position information**: Added via positional encodings since attention is permutation-invariant.

---

## See Also
- [[Attention Mask]]
- [[Understanding Projections in Self-Attention (Q, K, V)]]
- [Vaswani et al., "Attention is All You Need" (2017)](https://arxiv.org/pdf/1706.03762)
