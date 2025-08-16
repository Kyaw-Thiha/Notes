# Deployment Frameworks 
 #llm/deployment #hugging-face/llm 
> Goal: Overview of common **LLM inference/deployment frameworks** and how to use them effectively.

---

## 1. **Text Generation Inference (TGI)**

**What it is**  
- Hugging Face’s official optimized inference server for text generation models.  
- Written in Rust & Python, supports Tensor Parallelism, quantization, batching, streaming.  
- Production-ready for large Transformer models.

**Why use it**  
- Turn any Hugging Face model into a scalable API endpoint.  
- Efficient serving with GPU acceleration and dynamic batching.  
- Ideal for enterprise-scale deployment.

**How to implement**  
- Install via Docker (recommended):
```bash
docker run --gpus all --shm-size 1g -p 8080:80 \
  -v $PWD:/data ghcr.io/huggingface/text-generation-inference:latest \
  --model-id meta-llama/Llama-2-7b-chat-hf
```

- Query via API:
```python
import requests

resp = requests.post("http://localhost:8080/generate", json={"inputs": "Hello"})
print(resp.json())
```

**Extra features**  
- Supports streaming tokens for chatbots.  
- Built-in quantization (bitsandbytes, GPTQ).  
- Optimized for GPUs, but can fall back to CPUs.

---

## 2. **vLLM**

**What it is**  
- High-throughput inference engine for LLMs.  
- Uses **PagedAttention** (efficient KV-cache management) to drastically improve throughput.  
- Popular for serving chatbots and long-context models.

**Why use it**  
- Extremely efficient on GPUs, especially for batched requests.  
- Handles long-context models (e.g., 16k+ tokens) better than many alternatives.  
- Integrates directly with Hugging Face and OpenAI-compatible APIs.

**How to implement**  
- Install:
```bash
pip install vllm
```

- Run a model as an API server:
```bash
python -m vllm.entrypoints.api_server \
    --model facebook/opt-6.7b \
    --tensor-parallel-size 2
```

- Query via Python:
```python
import requests

resp = requests.post("http://localhost:8000/generate", json={"prompt": "Hello world"})
print(resp.json())
```

**Extra features**  
- Drop-in replacement for OpenAI API (chat/completions format).  
- Great for research + production with GPU clusters.  
- Focused on **maximizing throughput** and **memory efficiency**.

---

## 3. **llama.cpp**

**What it is**  
- Lightweight C++ inference engine for running LLaMA and other HF-converted models locally.  
- Uses CPU and GPU acceleration (via Metal, CUDA, OpenCL).  
- Famous for enabling LLMs on laptops, desktops, even phones.

**Why use it**  
- No heavyweight dependencies.  
- Runs quantized models (down to 4-bit) on commodity hardware.  
- Perfect for **edge devices**, personal experimentation, or low-cost deployment.

**How to implement**  
- Build from source:
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make
```

- Convert HF model to GGUF format (new standard):
```bash
python convert.py --model llama-2-7b-hf --outfile llama-2-7b.gguf
```

- Run inference:
```bash
./main -m llama-2-7b.gguf -p "Hello world"
```

**Extra features**  
- Supports quantization levels: q4, q5, q8.  
- Can run on CPUs, Macs (Metal), GPUs (CUDA).  
- Embedded-friendly (Raspberry Pi, Jetson).  

---

## Quick Comparison

| Framework   | Best For | Hardware | Strengths |
|-------------|----------|----------|------------|
| **TGI**     | Enterprise API serving | GPU clusters | Scalable, production-ready, streaming, Hugging Face integration |
| **vLLM**    | High-throughput GPU inference | GPUs | PagedAttention, long context, OpenAI-compatible API |
| **llama.cpp** | Local & edge inference | CPU/GPU (lightweight) | Runs quantized models on commodity devices, minimal setup |

---

## Key Takeaways

- **TGI** = enterprise-grade, feature-rich Hugging Face serving solution.  
- **vLLM** = research+production engine with best GPU throughput.  
- **llama.cpp** = lightweight C++ for running quantized models anywhere, even offline.  

When choosing:
- For **production APIs** → TGI.  
- For **max throughput on GPU clusters** → vLLM.  
- For **personal/edge device use** → llama.cpp.  

---
