#llm/inference
# Inference Process
Large Language Models (LLMs) generate text using a two-step inference process:
1. **Prefill Phase**
2. **Decode Phase**

Understanding these phases is essential for optimizing performance and grasping how autoregressive models like GPT work during inference.

---

## 🔹 Prefill Phase

### 📖 Description
The **prefill** phase is the model's initial pass through the *entire input sequence*. This is equivalent to "reading the question before answering."

It includes:

- **Tokenization:** Converts raw text into token IDs using a tokenizer (e.g., Byte Pair Encoding).
- **Embedding Conversion:** Maps each token ID to a vector from the embedding matrix:
  
  $$ x_i = \text{Embedding}(t_i) \in \mathbb{R}^d $$

- **Initial Processing:** These embeddings are fed through all Transformer layers to produce a *contextual representation*:

  $$
  h_i^L = \text{TransformerLayers}(x_1, x_2, \dots, x_n)
  $$

### 💡 Characteristics
- **Parallelizable:** All input tokens are processed at once.
- **Computationally expensive:** Especially with long inputs.
- **Output:** Final hidden states for each token — cached for later decoding.

---

## 🔹 Decode Phase

### 📖 Description
Once the initial input is processed, the **decode** phase begins. Here, the model **generates one token at a time** using an **autoregressive** process.

Each iteration includes:

1. **Attention Computation:** Attend to all previous tokens.
2. **Logit Computation:**
   $$
   \text{logits}_{i} = W_o \cdot h_i^L
   $$
3. **Probability Calculation:**
   $$
   P(t_i | t_1, \dots, t_{i-1}) = \text{softmax}(\text{logits}_i)
   $$
4. **Token Selection:** Greedy / Sampling / Top-k / Top-p.
5. **Stopping Condition:** Based on max tokens, EOS token, etc.

### 💡 Characteristics
- **Autoregressive:** New token depends on all previous tokens.
- **Memory-intensive:** Must cache and re-use key/value pairs from self-attention.
- **Not parallelizable per step**, but can batch multiple sequences.

---

## 🔁 Putting It All Together

| Phase       | Input              | Output                     | Parallel?      | Notes                               |
|-------------|--------------------|-----------------------------|----------------|-------------------------------------|
| Prefill     | Prompt (t₁, ..., tₙ) | Hidden states + KV Cache     | ✅ Yes         | One-time cost per prompt            |
| Decode      | Previous tokens    | Next token                  | ❌ No (per step) | Iterative, can be long              |

---

## 🧠 Attention & KV Cache

During decoding, **key** and **value** vectors from each layer are **cached** for efficiency.

- **Why?** Reduces redundant computation.
- Instead of reprocessing the full context, only the new token’s representations are computed and appended.
- This is often implemented with:
  $$
  \text{cache}_{\text{key}} \leftarrow [K_1, K_2, \dots, K_{t-1}]
  $$

---

## 🧩 Autoregressive vs Encoder-Decoder

| Model Type         | Prefill Phase | Decode Phase |
|--------------------|---------------|--------------|
| Decoder-only (GPT) | Full context  | Yes (token-by-token) |
| Encoder-decoder (T5, BART) | Input → Encoder | Decoder runs with encoder output |
| Encoder-only (BERT) | Full context | ❌ No decoding |

Decoder-only models require both prefill and decode. Encoder-only models use just a single full pass (no autoregression).

---

## ⚙️ Optimization Techniques

- **KV Caching:** Speeds up decode phase.
- **CUDA Graphs / TensorRT / ONNX:** Optimize inference path.
- **Speculative Decoding:** Use small model to predict a few tokens, then verify with large model.

---

## 📌 Summary

- The two-phase process allows LLMs to generate coherent responses with full context awareness.
- The **prefill phase** processes the prompt as a whole. It’s compute-heavy but one-time.
- The **decode phase** produces one token at a time based on prior output. It’s memory-heavy and iterative.
- Understanding this division is essential for debugging slow inference, optimizing latency, and deploying real-time systems.

---
## 🔗 See Also

- [[Self-Attention]]
- [[Decoder]]
- [[Decoder-Only Model]]
- [[KV Cache]]
- [[Sampling Strategies]]
