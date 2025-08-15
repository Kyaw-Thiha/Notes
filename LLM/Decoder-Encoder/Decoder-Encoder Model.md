#ml #llm/transformers/decoder-encoder 
# Encoder-Decoder Models 

## 🧠 What is an Encoder-Decoder Model?

An **Encoder-Decoder** model maps an input sequence to an output sequence using two neural components:

- The **encoder** converts the input into hidden representations
- The **decoder** generates the output using those representations

---

## 📦 Use Cases

- 🈂️ Machine Translation (e.g., English → Japanese)
- 📝 Text Summarization
- 🎤 Speech-to-Text
- 🖼️ Image Captioning
- 💻 Code Generation

---

## 🧮 Mathematical Formulation

Let:
- Input: $X = (x_1, x_2, ..., x_T)$
- Output: $Y = (y_1, y_2, ..., y_{T'})$
- Encoder hidden states: $H = (h_1, h_2, ..., h_T)$

### Encoder:
$$
h_t = \text{EncoderLayer}(x_t)
$$

### Decoder (autoregressive):
$$
s_t = \text{DecoderLayer}(y_{t-1}, H)
$$
$$
P(y_t | y_{<t}, X) = \text{Softmax}(W_o s_t)
$$

---

## 🔢 Numerical Matrix Example 

Assume:

- Input tokens: "I love"
- Token embeddings (dimension 2):

$$
x_1 = [1, 0], \quad x_2 = [0, 1]
$$

### Encoder:
Let encoder be a linear layer:
$$
W_{enc} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
$$

Then:
$$
h_1 = W_{enc} \cdot x_1 = \begin{bmatrix} 1 \\ 3 \end{bmatrix} \\
h_2 = W_{enc} \cdot x_2 = \begin{bmatrix} 2 \\ 4 \end{bmatrix}
$$

Final encoder output: $H = [h_1, h_2] = [[1, 3], [2, 4]]$

---

### Decoder:
Let decoder take previous token and compute:

$$
W_{dec} = \begin{bmatrix} 1 & -1 \\ 0 & 1 \end{bmatrix}
$$

Let $y_0 = [0.5, 0.5]$ (e.g., "<sos>" token):

$$
s_1 = W_{dec} \cdot y_0 = [0, 0.5]
$$

Then attention to encoder:
$$
\text{Attention}(s_1, H) = \text{softmax}(s_1 \cdot H^T) \cdot H
$$

Compute dot products:
- $s_1 \cdot h_1 = 0*1 + 0.5*3 = 1.5$
- $s_1 \cdot h_2 = 0*2 + 0.5*4 = 2$

Softmax over $[1.5, 2]$:
$$
\text{softmax}([1.5, 2]) = \left[\frac{e^{1.5}}{e^{1.5}+e^2}, \frac{e^2}{e^{1.5}+e^2}\right] \approx [0.38, 0.62]
$$

Weighted sum:
$$
c = 0.38 \cdot [1, 3] + 0.62 \cdot [2, 4] = [1.62, 3.62]
$$

This $c$ is the **context vector** the decoder uses to generate the first output token.

---

## 🧪 Minimal Encoder-Decoder (No Library)

```python
import math

# Dummy embedding vectors
x1 = [1, 0]
x2 = [0, 1]
y0 = [0.5, 0.5]

# Linear layer matrices
W_enc = [[1, 2], [3, 4]]
W_dec = [[1, -1], [0, 1]]

def matmul(W, v):
    return [sum(wi * vi for wi, vi in zip(row, v)) for row in W]

def dot(v1, v2):
    return sum(a * b for a, b in zip(v1, v2))

def softmax(xs):
    exps = [math.exp(x) for x in xs]
    total = sum(exps)
    return [x / total for x in exps]

def weighted_sum(weights, vectors):
    return [sum(w * v[i] for w, v in zip(weights, vectors)) for i in range(len(vectors[0]))]

# Encode
h1 = matmul(W_enc, x1)
h2 = matmul(W_enc, x2)
H = [h1, h2]

# Decode
s1 = matmul(W_dec, y0)

# Attention
attn_scores = [dot(s1, h) for h in H]
attn_weights = softmax(attn_scores)
context = weighted_sum(attn_weights, H)

print("Encoder outputs:", H)
print("Decoder state:", s1)
print("Attention weights:", attn_weights)
print("Context vector:", context)
```

# 🧠 Summary
This example shows how encoder transforms inputs and decoder uses attention to focus on important parts of the input.

Real models are much deeper and use multi-head attention, layer norms, and positional encoding.

But the core flow remains the same: Encode → Attend → Decode.

## See Also
- [[Encoder-Only Model]]
- [[Decoder-Only Model]]