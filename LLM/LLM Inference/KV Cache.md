#llm/inference/kv-cache 
# 🧠 KV Cache in LLM Inference

KV (Key-Value) caching is a core optimization that dramatically improves the speed and efficiency of **decoder-only language models** like GPT. 

This technique enables fast token-by-token generation while managing memory and compute constraints in long sequences.

You can find the code examples at
- [[KV Cache (Code)#In Python (without libraries)|Simple Example]]
- [[KV Cache (Code)#Real World KV Cache (with library)|Real-World Example]]

---

## 🧩 What Is KV Caching?

During text generation, the Transformer decoder computes **self-attention** over all previously generated tokens.

At each step:
- The model needs to **attend to every past token**
- This becomes **slower** as the context grows

### 💡 Insight:
Most of the previous computations (keys and values) **don’t change** — so we can cache and reuse them!

---

## 🧮 Mathematical View

For each token $t_i$ at layer $l$:
- Compute query $Q_i^l$, key $K_i^l$, and value $V_i^l$

$$
\text{Attention}(Q_i^l, K_{\leq i}^l, V_{\leq i}^l) = \text{softmax}\left(\frac{Q_i^l {K_{\leq i}^l}^\top}{\sqrt{d_k}} \right) V_{\leq i}^l
$$

Instead of recomputing $K_j^l$ and $V_j^l$ for $j = 1, \dots, i-1$ at every step, **KV caching** stores:

- $K_{\leq i-1}^l = [K_1^l, \dots, K_{i-1}^l]$
- $V_{\leq i-1}^l = [V_1^l, \dots, V_{i-1}^l]$

At step $i$, we only compute $K_i^l$, $V_i^l$ and **append** them to the cache.

---

## 🚀 Benefits of KV Caching

| Benefit                     | Explanation |
|----------------------------|-------------|
| ⏱️ Faster Decoding         | Avoids recomputation of past keys/values. |
| 🧠 Memory-Time Tradeoff     | Uses more memory (for the cache), but drastically speeds up decoding. |
| 📏 Enables Long Contexts    | Without caching, large contexts are impractical to decode efficiently. |

---

## 📊 Key Performance Metrics

| Metric                     | Description |
|----------------------------|-------------|
| **TTFT** (Time to First Token) | Measures latency of the **prefill phase** (processing the prompt). |
| **TPOT** (Time per Output Token) | Measures speed of the **decode phase** (affected by caching). |
| **Throughput**             | Number of requests per second across users (affects scalability). |
| **VRAM Usage**             | Total GPU memory consumed — cache grows with context length and model size. |

> KV caching optimizes TPOT — **essential** for responsive, real-time applications.

---

## 📦 VRAM vs. Speed Tradeoff

- **Without cache:** You reprocess the entire context for every token.
- **With cache:** You store $\mathcal{O}(L \cdot d)$ memory per layer — one vector per token, per layer.

| Factor         | Without KV Cache              | With KV Cache                  |
|----------------|-------------------------------|--------------------------------|
| Time Complexity | $O(L^2)$ per token            | $O(L)$ total (for $L$ tokens)  |
| Memory Usage    | Lower                         | Higher (key & value per token) |
| Decode Latency  | High                          | Low                            |

---

## ⚠️ Context Length Challenges

LLMs scale poorly with longer prompts:

- **Attention memory cost:** $\mathcal{O}(L^2)$ per layer if not cached
- **Prefill dominates TTFT**: For long prompts, response delay increases

> **KV caching mitigates this by shifting cost away from decoding.**

---

## 🛠️ Advanced Optimizations

- **Sliding Window Attention:** Limit how far back tokens can attend (used in Longformer, GPT-4 Turbo).
- **Grouped Attention Heads:** Share KV cache across heads to reduce memory.
- **Quantized Caches:** Store KV tensors at reduced precision (e.g., FP8/INT4).

---

## 🧠 Takeaways

- KV caching is **essential** for efficient autoregressive generation.
- It turns long-context decoding from impractical to feasible.
- Tradeoff: Higher VRAM usage in exchange for **dramatic** speedups.

---

## 🔗 Related Concepts

- [[Inference Process]]
- [[Sampling Strategies]]
- [[KV Cache (Code)]]
- [[Decoder]]
- [[Self-Attention]]
