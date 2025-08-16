# Layer Dropping (Early Exit)
#llm/inference/optimization/layer-dropping  
## Overview
Large Language Models (LLMs) like GPT, BERT, and their variants are composed of many transformer layers (sometimes 24, 48, or more).  
Traditionally, **all tokens** are processed through **all layers**, which ensures maximum accuracy but is computationally expensive.  

**Layer dropping (early exit)** is an **inference optimization technique** where the model does not always use all layers. Instead, it may **exit early** once it has achieved sufficient confidence in its predictions.

---

## Motivation
- Many tokens do not need the full depth of the model for correct prediction.  
- Later layers often **refine** or **reconfirm** what earlier layers already predicted.  
- Adaptive compute: allocate more compute for *hard tokens*, less for *easy tokens*.  

This allows faster inference while preserving accuracy for most cases.

---

## How Early Exit Works

1. **Intermediate Classifiers ("Exit Heads")**  
   - Additional classifiers are attached at certain layers of the transformer (e.g., after layer 12, 24, 36 in a 48-layer model).  
   - These classifiers can generate partial predictions without waiting for the final layer.

2. **Confidence Estimation**  
   - After an exit head produces logits, the model estimates **confidence** in its prediction.  
   - Confidence can be measured using:
     - **Entropy of the softmax distribution** (lower entropy = higher confidence).
     - **Margin between top-1 and top-2 logits** (larger margin = higher confidence).
     - A **learned classifier** trained to predict if an early exit is safe.

3. **Decision Rule**  
   - If confidence > threshold → **stop early** and output prediction.  
   - Otherwise → continue to the next layer and repeat.  

4. **Final Guarantee**  
   - If no earlier exit passes the threshold, the model always produces a prediction at the last layer.

---

## Example Illustration

Imagine a 24-layer transformer:

- At layer 8: The exit head predicts "cat" with **95% confidence**. → Model exits here.  
- At layer 8 for another input: Confidence is only **55%**. → Continue deeper.  
- At layer 16: Confidence rises to **90%**. → Model exits here.  
- If confidence never exceeds threshold → Run until **layer 24**.

---

## Implementation Outline (PyTorch)

```python
import torch
import torch.nn.functional as F

def early_exit(logits, threshold=0.9):
    # logits: [batch, vocab_size]
    probs = F.softmax(logits, dim=-1)
    confidence, _ = torch.max(probs, dim=-1)
    return confidence > threshold

# Example loop through transformer layers
for layer_idx, layer in enumerate(transformer_layers):
    hidden_states = layer(hidden_states)
    
    if layer_idx in exit_layers:  # e.g., {8, 16, 24}
        logits = classifier_heads[layer_idx](hidden_states)
        if early_exit(logits, threshold=0.9):
            return logits  # Exit early
```

## Benefits
- Reduced inference time: Saves computation for easy tokens.
- Dynamic adaptation: Harder tokens can still leverage full depth.
- Energy efficiency: Less computation → lower energy usage.

## Trade-Offs
- Accuracy drop: Exiting too early may harm prediction quality.
- Overhead: Requires adding and training intermediate classifiers.
- Calibration needed: The confidence threshold must be tuned to balance speed vs accuracy.

## Practical Usage
- Works well in real-time applications (e.g., chatbots, translation) where latency is critical.
- Combined with other optimizations (quantization, pruning) for maximum efficiency.
- Research shows up to 30–50% faster inference with minimal accuracy loss if thresholds are tuned well.

## Summary
- Layer dropping / early exit enables adaptive compute for LLMs.
- Relies on intermediate classifiers and confidence-based stopping.
- Provides a practical way to balance speed and accuracy during inference.

## See Also
- [[Inference Optimization]]