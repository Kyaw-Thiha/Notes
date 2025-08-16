# Pruning in LLM Inference Optimization

## Overview
Pruning is a **model compression technique** where unnecessary parameters or structures are removed from a neural network to reduce its computational and memory footprint.  
In the context of **Large Language Models (LLMs)**, pruning helps optimize inference by reducing latency, memory usage, and energy consumption.

Unlike **layer dropping / early exit**, which dynamically skips computation at runtime, **pruning modifies the model structure itself** so that inference is permanently cheaper.

---

## Motivation
- LLMs contain **billions of parameters**, but not all contribute equally.  
- Many weights, neurons, or attention heads are **redundant** or **underutilized**.  
- Removing these elements can lead to:
  - Faster inference  
  - Smaller memory footprint  
  - Energy savings  

---

## Types of Pruning

### 1. Weight Pruning
- Removes individual weights (usually those with small magnitudes).  
- Results in **sparse matrices**, which can speed up inference if the hardware supports efficient sparse operations.  
- Example: Zeroing out 80% of smallest weights.

### 2. Neuron Pruning
- Removes entire **neurons** in the feed-forward (MLP) layers.  
- More structured than weight pruning → easier for hardware acceleration.  
- Example: Drop neurons that consistently have low activations.

### 3. Attention Head Pruning
- Transformers have multiple attention heads per layer.  
- Some heads capture redundant information and can be removed.  
- Example: Removing heads that consistently produce near-zero or uninformative attention maps.

### 4. Structured Pruning
- Removes entire **dimensions, blocks, or layers** of the model.  
- Produces a smaller, dense model that runs faster on standard hardware.  
- Example: Pruning hidden dimensions in MLPs by 25%.

---

## How Pruning Works

1. **Identify unimportant parameters**
   - Based on weight magnitude (e.g., L1 norm).  
   - Based on gradients or sensitivity analysis.  
   - Learned importance scores (using an auxiliary model).

2. **Remove them**
   - Set them to zero (unstructured pruning).  
   - Physically remove neurons/heads (structured pruning).

3. **Fine-tune the model**
   - After pruning, the model often loses accuracy.  
   - Fine-tuning helps the smaller model regain performance.  

---

## Example: PyTorch Pruning

```python
import torch
import torch.nn.utils.prune as prune

# Example: Pruning 30% of weights in a linear layer
linear = torch.nn.Linear(512, 512)

prune.l1_unstructured(linear, name="weight", amount=0.3)

print(list(linear.named_parameters()))  # includes pruned weights
print(list(linear.named_buffers()))     # includes mask
```

- `prune.l1_unstructured` zeroes out the smallest 30% of weights.
- Structured methods like `prune.ln_structured` can prune entire neurons or channels.

## Example: Attention Head Pruning (Conceptual)
```python
def prune_attention_heads(layer, heads_to_prune):
    # heads_to_prune = [0, 2] → remove heads 0 and 2
    mask = torch.ones(layer.num_heads)
    mask[heads_to_prune] = 0
    layer.head_mask = mask
    return layer
```
This reduces compute during self-attention since fewer heads are evaluated.

## Benefits
- Inference speedup (especially with structured pruning).
- Memory savings → lower VRAM usage.
- Energy efficiency.
- Smaller deployment footprint (useful for edge devices).

## Trade-Offs
- Accuracy drop if too much is pruned.
- Unstructured pruning (zeroing weights) requires specialized hardware to see real speedups.
- Fine-tuning cost: After pruning, extra training is needed.

# Practical Usage
Often combined with quantization and early exit for maximum gains.

Works best in:
- Deployment on edge devices (phones, IoT, embedded boards).
- Large-scale server inference where efficiency is critical.
- Research shows that pruning up to 40–60% of parameters can often be done with minimal accuracy loss if done carefully with fine-tuning.

## Summary
- Pruning reduces the model size and compute permanently by removing unnecessary weights, neurons, or attention heads.
- Comes in different forms: weight, neuron, attention head, structured pruning.
- Requires fine-tuning to recover lost accuracy.
- A fundamental tool for LLM inference optimization alongside early exit and quantization.

## See Also
- [[Quantization]]
- [[Inference Optimization]]