# Caching 
 #llm/inference/optimization/caching

## Overview
**Caching** is a key technique for speeding up **Large Language Model (LLM) inference**, especially in **autoregressive decoding** (e.g., GPT-style text generation).  
The idea is to **reuse intermediate computations** instead of recomputing them at every step.  

Without caching, generating a sequence of length *N* requires recomputing all attention values for all *N* tokens at each decoding step.  
With caching, previously computed key-value pairs are stored and reused, reducing computation dramatically.

---

## Motivation
- LLMs generate text **one token at a time**.  
- At each decoding step, the model attends to **all past tokens**.  
- Naive approach → quadratic cost in sequence length.  
- With **KV caching**:
  - Store **key (K)** and **value (V)** matrices from past steps.  
  - Reuse them instead of recomputing.  
  - Only compute attention against **the new token**.  

This reduces per-step complexity from **O(n²)** to **O(n)**.

---

## Types of Caching in LLMs

### 1. Key-Value (KV) Cache
- Stores the **K and V matrices** for each token from previous steps.  
- At step *t*, only the new token’s K and V are computed.  
- The cache is concatenated to form the full sequence context.  

---

### 2. Attention Cache
- General term for storing intermediate attention states.  
- Can be optimized with **memory layouts** like:
  - **PagedAttention (vLLM)** → manages KV cache in paged memory blocks.  
  - **FlashAttention** → avoids redundant memory reads/writes by fusing attention computation.  

---

### 3. Layer Output Cache
- Stores outputs of intermediate layers to avoid recomputation if needed.  
- Useful in **beam search** or **speculative decoding**.  

---

## How KV Caching Works

1. During generation, at each decoding step:
   - Compute Q, K, V for the **new token**.  
   - Append K, V to cache.  
   - Use Q to attend over all cached K, V.  

2. Pseudocode (PyTorch-style):

```python
# cache = {"k": [layer1_k, ...], "v": [layer1_v, ...]}

def forward_step(model, token, cache):
    new_k, new_v = [], []
    for layer, (k_cache, v_cache) in zip(model.layers, zip(cache["k"], cache["v"])):
        q, k, v = layer.self_attn(token)
        k_cache = torch.cat([k_cache, k], dim=1)
        v_cache = torch.cat([v_cache, v], dim=1)
        token = layer.feed_forward(layer.attend(q, k_cache, v_cache))
        new_k.append(k_cache)
        new_v.append(v_cache)
    return token, {"k": new_k, "v": new_v}
```

---

## Advanced Caching Optimizations

### 1. PagedAttention (vLLM)
- Memory-efficient KV cache layout.  
- Stores cache in **fixed-size memory pages**.  
- Avoids costly memory copies when sequences grow.  
- Enables **continuous batching** with dynamic sequence lengths.  

### 2. FlashAttention
- Fuses attention computation into a single kernel.  
- Reduces memory traffic for cache reads/writes.  
- Supports efficient KV cache usage with large sequence lengths.  

### 3. Cache Quantization
- Store KV cache in **lower precision** (e.g., FP16, INT8).  
- Saves memory and bandwidth with little accuracy loss.  

### 4. Cache Sharing for Beam Search
- In beam search, multiple beams share a common prefix.  
- Shared KV caches prevent recomputation of the same prefix multiple times.  

---

## Benefits
- **Drastic speedup** in autoregressive decoding (up to 10×).  
- **Reduced memory bandwidth** usage.  
- Enables **long-sequence inference**.  
- Crucial for **real-time applications** like chatbots.  

---

## Trade-Offs
- **Memory footprint** grows with sequence length (KV cache must store all tokens).  
- **Batching complexity** increases (different requests have different cache sizes).  
- Requires specialized implementations for **continuous batching**.  
- **Quantizing cache** may slightly reduce accuracy.  

---

## Practical Usage
- Most modern inference frameworks support caching:
  - **Hugging Face Transformers** → KV caching enabled in generation APIs.  
  - **vLLM** → PagedAttention for efficient caching.  
  - **DeepSpeed-Inference** → optimized KV cache across GPUs.  
  - **TensorRT-LLM** → optimized GPU kernels for KV cache.  

---

## Summary
- **Caching is essential** for efficient LLM inference.  
- **KV cache** stores past key/value pairs to avoid recomputation.  
- Advanced methods like **PagedAttention** and **FlashAttention** improve cache efficiency.  
- Enables **low-latency, high-throughput text generation** at scale.  

---
## See Also
- [[Inference Optimization]]
- [[KV Cache]]
- [[Quantization]]
- [[Decoder-Only Model]]