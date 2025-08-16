# Quantization 
#llm/inference/optimization/quantization  
## Overview
Quantization is a technique to **reduce the numerical precision** of a model’s parameters and/or activations.  
Instead of storing and computing with 32-bit floating point numbers (FP32), we use **lower precision formats** like FP16, INT8, or even INT4.  

This reduces:
- **Memory usage** (smaller weights).  
- **Computation cost** (faster matrix multiplications).  
- **Energy consumption**.  

Quantization is one of the most widely used methods to make **Large Language Models (LLMs)** feasible for deployment.

---
## Motivation
- LLMs contain **billions of parameters**.  
- At FP32, a 7B parameter model requires **28 GB of memory** (7B × 4 bytes).  
- Reducing precision allows the same model to run on much smaller GPUs or even edge devices.  

---
## Types of Quantization

### 1. Post-Training Quantization (PTQ)
- Apply quantization **after training**.  
- No retraining needed → faster, but may reduce accuracy.  
- Works best with simple precision reduction (e.g., FP32 → INT8).  

### 2. Quantization-Aware Training (QAT)
- Train the model while simulating quantization effects.  
- Model learns to be robust against lower precision.  
- More accurate than PTQ, but requires retraining.  

### 3. Dynamic vs Static Quantization
- **Dynamic**: Weights quantized before inference; activations quantized on-the-fly. Fast, but less optimized.  
- **Static**: Both weights and activations are quantized using calibration data. More optimized for inference.  

### 4. Mixed-Precision Quantization
- Use **different precisions** for different parts of the model.  
  - Example: Use INT8 for weights, FP16 for attention softmax.  
- Balances speed and accuracy.  

---

## Common Precisions
- **FP32**: Standard training precision.  
- **FP16 (Half precision)**: Widely used for training & inference (saves 50% memory).  
- **BF16 (Brain Float 16)**: Similar to FP16, but more stable for training.  
- **INT8**: Most common quantization for inference, ~4× smaller than FP32.  
- **INT4 / INT2**: Aggressive quantization for very large models; higher risk of accuracy loss.  

---

## Example: Quantization in PyTorch

```python
import torch
from torch.ao.quantization import quantize_dynamic

# Example: Quantizing a transformer model
import transformers
model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")

# Apply dynamic INT8 quantization to linear layers
quantized_model = quantize_dynamic(
    model, 
    {torch.nn.Linear}, 
    dtype=torch.qint8
)

print(quantized_model)
```

This reduces the model size and speeds up inference while retaining most accuracy.

---

## Example: LLM.int8() (Hugging Face Accelerate)

```python
from transformers import AutoModelForCausalLM

# Load GPT with 8-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    load_in_8bit=True,
    device_map="auto"
)
```

This uses **bitsandbytes** backend for efficient INT8 inference.

---

## Benefits
- **4× smaller models** with INT8 compared to FP32.  
- **Speedups** on CPUs and GPUs that support low-precision math (e.g., NVIDIA Tensor Cores).  
- **Lower energy consumption** → cheaper large-scale deployment.  
- Enables deployment of **huge models on smaller hardware**.  

---

## Trade-Offs
- **Accuracy drop** if precision is too low (especially INT4 or INT2).  
- **Hardware dependency**: Not all devices accelerate low-precision math equally.  
- **Extra complexity**: Calibration and fine-tuning may be required.  

---

## Practical Usage
- **Server-side inference**: INT8 quantization is standard.  
- **Edge deployment**: INT4 or mixed precision (for small devices).  
- **Training**: Often uses FP16 or BF16 for speed + stability.  

Many production systems use **hybrid setups**:  
- FP16 for attention softmax.  
- INT8 for linear layers.  
- FP32 for final logits.  

---

## Summary
- **Quantization reduces model precision → faster, smaller, cheaper inference.**  
- Key methods: **Post-Training Quantization (PTQ)** and **Quantization-Aware Training (QAT)**.  
- Widely used in LLM deployment, often combined with **pruning** and **early exit**.  
- Provides one of the best trade-offs between performance and efficiency in practice.  

---
## See Also
- [[Inference Optimization]]