#ml #llm/transformers/feed-forward-neural-network 
# 🧠 Feed-Forward Neural Network in Transformers

In the Transformer architecture, a **feed-forward neural network (FFN)** is applied **independently to each token** after the self-attention (and cross-attention in decoders). 
This component brings **non-linearity**, **feature transformation**, and **modeling capacity** to the system.

---

## 📐 Structure

Each token embedding $x \in \mathbb{R}^{d_{\text{model}}}$ is passed through a **2-layer MLP** with a ReLU activation:

$$
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

Where:
- $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$
- $W_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$
- $b_1 \in \mathbb{R}^{d_{\text{ff}}}$, $b_2 \in \mathbb{R}^{d_{\text{model}}}$
- $\max(0, \cdot)$ is the ReLU activation

This transformation is applied to each position (token) **separately but with shared parameters**.

---

## 🔢 Mathematical Example

Let:
- $d_{\text{model}} = 4$, $d_{\text{ff}} = 8$
- A token vector:  
  $$ x = \begin{bmatrix} 1 \\ 2 \\ -1 \\ 0 \end{bmatrix} $$

Let $W_1$ and $W_2$ be initialized with small values, and $b_1 = b_2 = 0$. First:

$$
h = \max\left(0, W_1^\top x + b_1\right) \in \mathbb{R}^8
$$

Then:

$$
\text{FFN}(x) = W_2^\top h + b_2 \in \mathbb{R}^4
$$

This output vector replaces the original $x$ in the residual connection.

---
## Code Example
**Minimal Example**
```python
ff = np.matmul(x, self.W1)
ff = np.maximum(0, ff)  # ReLU
ff = np.matmul(ff, self.W2)
```

**Pytorch Example**
```python
# PyTorch-style FFN module
nn.Sequential(
    nn.Linear(d_model, d_ff),
    nn.ReLU(),
    nn.Linear(d_ff, d_model)
)
```
---

## 🛠 Characteristics

- Applied **position-wise** (same MLP for each token)
- **Non-sequential**: no dependence across tokens
- Usually, $d_{\text{ff}} = 2048$ when $d_{\text{model}} = 512$
- Often wrapped in **residual + layer normalization**:

$$
\text{Output} = \text{LayerNorm}(x + \text{FFN}(x))
$$

---
## ✅ Summary

- FFNs enable **per-token transformation**
- No attention or sequence interaction happens here
- Efficient and parallelizable
- Essential for learning rich token representations



