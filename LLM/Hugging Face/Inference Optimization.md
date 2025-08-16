# Inference Optimization 

 #llm/inference/optimization #hugging-face/llm  

## Quick Summary

 - **[[#1. Quantization]]** → smaller precision, faster compute.  
- **[[#2. Distillation]]** → smaller student mimicking large teacher.  
- **[[#3. Pruning]]** → remove redundant heads/weights.  
- **[[#4. Layer Dropping / Early Exit]]** → fewer layers at inference.  
- **[[#5. Hardware Acceleration (ONNX, TorchScript, etc.) |5. Hardware Acceleration]]** → ONNX/TorchScript/TensorRT.  
- **[[#6. Batch Inference]]** → throughput boost.  
- **[[#7. Caching (for Decoders) |7. Caching]]** → speed up autoregressive decoding.
---

## 1. Quantization

**How it works**  
- Converts model weights (normally FP32) to lower precision (e.g., FP16, INT8, INT4).  
- Reduces memory footprint, speeds up matrix multiplications, sometimes with minor accuracy loss.  
- Useful when deploying on CPU or constrained GPU.

**Implementation**  
```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english",
    device_map="auto",
    load_in_8bit=True   # Requires bitsandbytes
)
```

Notes:
- `load_in_8bit=True` → 8-bit quantization (needs `bitsandbytes`).
- `load_in_4bit=True` → 4-bit (even smaller, but more accuracy tradeoff).
- Can also combine with training via **QLoRA** for fine-tuning.

[[Quantization|Read More]]

---

## 2. Distillation

**How it works**  
- Train a **smaller student model** to mimic the outputs (soft labels) of a **larger teacher model**.  
- Student learns both from real labels and teacher’s probability distribution.  
- Preserves most accuracy but with fewer parameters → faster inference.

**Implementation**  
- Hugging Face provides pre-distilled models like **DistilBERT** (a distilled version of BERT).  
- Training flow:
  1. Run teacher model to generate logits on dataset.
  2. Train student with a loss combining:
     - Standard cross-entropy (true labels).
     - KL divergence with teacher logits (soft targets).
- Example (conceptual):
```python
# Using a distilled checkpoint directly
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)
```

[[Knowledge Distillation|Read More]]

---

## 3. Pruning

**How it works**  
- Remove redundant weights or attention heads that contribute little to predictions.  
- Leads to sparse models that run faster, especially on hardware supporting sparse matmuls.  
- Can be structured (remove entire heads/layers) or unstructured (individual weights).

**Implementation**  
- Hugging Face supports **pruning attention heads**:
```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
# Remove heads from certain layers
model.prune_heads({
    0: [1, 2],   # remove heads 1,2 from layer 0
    1: [0]       # remove head 0 from layer 1
})
```
[[Pruning|Read More]]

---

## 4. Layer Dropping / Early Exit

**How it works**  
- During inference, not all inputs need the full depth of the model.  
- **Layer dropping**: Skip some Transformer layers at runtime (static).  
- **Early exit**: Dynamically stop once confidence is high enough.  
- Tradeoff: lower latency, potentially lower accuracy.

**Implementation**  
- Some distilled models already drop layers (e.g., DistilBERT has 6 instead of 12).  
- For dynamic early exit: use models fine-tuned with an **exit strategy** (not in all Hugging Face checkpoints).  
- Example of loading a distilled checkpoint (already layer-dropped):
```python
from transformers import AutoModel

# DistilBERT has 6 layers instead of BERT's 12
model = AutoModel.from_pretrained("distilbert-base-uncased")
```

[[Layer Dropping (Early Exit)|Read More]]

---

## 5. Hardware Acceleration (ONNX, TorchScript, etc.)

**How it works**  
- Convert models to formats optimized for inference engines:
  - **ONNX Runtime** → cross-platform, graph-level optimization.
  - **TorchScript** → optimized PyTorch graph for deployment.
  - **TensorRT** (NVIDIA) → GPU-accelerated kernels.

**Implementation**  
- Export to ONNX:
```bash
python -m transformers.onnx --model=distilbert-base-uncased onnx_model/
```
- Run with ONNX Runtime:
```python
import onnxruntime as ort

session = ort.InferenceSession("onnx_model/model.onnx")
```
- TorchScript export:
```python
traced = torch.jit.trace(model, (batch["input_ids"], batch["attention_mask"]))
torch.jit.save(traced, "traced.pt")
```

Read more at
- [[Hardware Optimization (General)]]
- [[Hardware Optimization (ONNX)]]
- [[Hardware Optimization (Torchscript)]]
- [[Hardware Optimization (TensorRt)]]

---

## 6. Batch Inference

**How it works**  
- Run multiple sequences in parallel instead of one-by-one.  
- Better GPU utilization, amortizes overhead.  
- Latency per request increases slightly, but throughput improves massively.

**Implementation**  
```python
from transformers import pipeline

pipe = pipeline("sentiment-analysis", batch_size=32)

texts = ["I love transformers!"] * 64
outputs = pipe(texts)  # processed in batches of 32
```
[[Batch Inference|Read More]]

---

## 7. Caching (for Decoders)

**How it works**  
- In autoregressive generation (GPT-like models), each new token requires recomputing attention.  
- **Caching** stores past key/value states → avoids recomputation → faster decoding.

**Implementation**  
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

inputs = tok("Hello", return_tensors="pt")
out = model.generate(
    **inputs, 
    max_new_tokens=20, 
    use_cache=True   # enables caching for faster decoding
)
```
[[Caching|Read More]]

---

## Quick Summary

- **Quantization** → smaller precision, faster compute.  
- **Distillation** → smaller student mimicking large teacher.  
- **Pruning** → remove redundant heads/weights.  
- **Layer dropping / Early exit** → fewer layers at inference.  
- **Hardware acceleration** → ONNX/TorchScript/TensorRT.  
- **Batching** → throughput boost.  
- **Caching** → speed up autoregressive decoding.

---
## See Also
- [[Quantization]]
- [[Knowledge Distillation]]
- [[Pruning]]
- [[Layer Dropping (Early Exit)]]
- [[Hardware Optimization (General)]]
- [[Hardware Optimization (ONNX)]]
- [[Hardware Optimization (Torchscript)]]
- [[Hardware Optimization (TensorRt)]]
- [[Batch Inference]]
- [[Caching]]