#ml #llm/transformers/projections 
# Understanding Projections in Self-Attention (Q, K, V)

This page answers common questions about **Query (Q)**, **Key (K)**, and **Value (V)** in self-attention:

---

## ✅ How are Input Embeddings Defined?

1. **Tokenization**: Input text is tokenized, e.g.  
`"I like cats"` → `["I", "like", "cats"]`

2. **Vocabulary Lookup**: Each token is assigned an ID, e.g.  
`["I", "like", "cats"]` → `[11, 87, 201]`

3. **Embedding Lookup**: These IDs map to **dense vectors** via a trainable embedding matrix $$E \in \mathbb{R}^{V \times d_{\text{model}}}$$ where $V$ is vocab size.

Example:
$$
x_1 = E[11], \quad x_2 = E[87], \quad x_3 = E[201]
$$

The embeddings matrix $X$ for the sequence is:
$$
X = \begin{bmatrix}
1 & 0 \\
0 & 1 \\
1 & 1
\end{bmatrix}
$$
where $d_{model}=2$

---

## ✅ Why Project to Q, K, V?

### Naive Question
> Why not just directly learn the attention weights instead of projecting to Q, K?

### Answer
- The projections:
$$
Q = XW^Q, \quad K = XW^K, \quad V = XW^V
$$
allow **learning relationships dynamically from input data**.  
- Directly parameterizing attention (without Q/K) would lead to **static attention**, fixed across all inputs.

### Mathematical Example

Let:
$$
W^Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad
W^K = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}
$$

Then:
$$
Q = XW^Q = X
$$
$$
K = XW^K
= \begin{bmatrix}
1 & 1 \\
0 & 1 \\
1 & 2
\end{bmatrix}
$$
Compute:
$$
QK^T
= \begin{bmatrix}
1 & 0 & 1 \\
1 & 1 & 2 \\
2 & 1 & 3
\end{bmatrix}
$$
Scaling by $$\frac{1}{\sqrt{d_k}} = \frac{1}{\sqrt{2}}$$ gives:
$$
\frac{QK^T}{\sqrt{2}} \approx
\begin{bmatrix}
0.71 & 0 & 0.71 \\
0.71 & 0.71 & 1.41 \\
1.41 & 0.71 & 2.12
\end{bmatrix}
$$

Apply softmax row-wise to get **attention weights**.

---

## ✅ Why Project to V?

> What if we didn't have $V$?

- $V$ contains the **content to be mixed together** after deciding attention via Q and K.
- Without $V$, you'd only have attention weights, **no content to blend**.

Example:
$$
W^V = \begin{bmatrix}
1 & 0 \\ 1 & 1
\end{bmatrix}
$$
$$
V = XW^V =
\begin{bmatrix}
1 & 0 \\
1 & 1 \\
2 & 1
\end{bmatrix}
$$

Then:
$$
\text{Attention Output} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

This yields contextual embeddings where **each token is a mix of others, weighted by learned relevance**.

---
## ✅ Diagram of Self-Attention Projection Flow
Input Embeddings (X)
│
├──> $W^Q$ ──> Q
│
├──> $W^K$ ──> K
│
└──> $W^V$ ──> V
   Q  ──────┐
    │       │
    ▼       │
  dot product: Q x $K^T$
    │
 Scaling by $\sqrt{ d_{k} }$
    │
 Softmax (attention weights)
    │
 Multiply with V
    │
Final Attention Output

---
## ✅ Summary

| Component | Purpose                                                 |
| --------- | ------------------------------------------------------- |
| Q, K      | Learn **where to attend** (relationship between tokens) |
| V         | Provides **what to attend** (content to blend)          |
This mechanism enables transformers to model **dynamic, input-specific relationships** between tokens.

---
## Related
- [[Self-Attention]]
- [[Attention Mask]]
