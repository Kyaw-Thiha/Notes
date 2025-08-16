# Knowledge Distillation 
 #llm/inference/optimization/knowledge-distillation  

## Overview
**Knowledge Distillation (KD)** is a model compression technique where a **smaller model (student)** learns to mimic the behavior of a **larger, well-trained model (teacher)**.  
The goal is to transfer the knowledge from the teacher to the student so that the student achieves comparable performance while being **smaller, faster, and more efficient** during inference.

---
## Motivation
- LLMs are extremely large (billions of parameters).  
- Running them in production or on edge devices is costly.  
- Distillation allows us to create **smaller models** that:
  - Retain much of the teacher’s performance.  
  - Are **faster to run**.  
  - Require **less memory**.  

Example: DistilBERT is ~40% smaller than BERT, but retains ~97% of its performance.

---
## How Knowledge Distillation Works

1. **Teacher Model**  
   - A large, pre-trained, accurate model.  
   - Provides **soft targets** (probability distributions) instead of just hard labels.

2. **Student Model**  
   - A smaller, simpler model (fewer layers, smaller hidden size).  
   - Trained to match the teacher’s output distributions.

3. **Loss Function**  
   - The student learns from two sources:
     1. **Hard loss**: Cross-entropy loss with ground truth labels.  
     2. **Soft loss**: Difference between student predictions and teacher’s soft predictions.  

   The combined loss is:

   $$
   L = \alpha \, L_{\text{hard}} + (1 - \alpha) \, L_{\text{soft}}
   $$

   where:  
   - $L_{\text{hard}}$ = cross-entropy with true labels.  
   - $L_{\text{soft}}$ = KL divergence between teacher and student outputs.  
   - $\alpha$ balances between the two.

4. **Temperature Scaling**  
   - Teacher’s logits are softened using a **temperature $T > 1$**:  

   $$
   p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
   $$

   - This makes the probability distribution less "peaked", giving the student richer information about class similarities.

---
## Example Training Loop (PyTorch)

```python
import torch
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.5):
    # Hard loss (true labels)
    hard_loss = F.cross_entropy(student_logits, labels)

    # Soft loss (teacher vs student)
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits / T, dim=-1),
        reduction="batchmean"
    ) * (T * T)

    return alpha * hard_loss + (1 - alpha) * soft_loss
```

This combines **hard labels** and **soft labels** for student training.

---
## Benefits
- **Smaller models** with nearly the same accuracy.  
- **Faster inference** due to fewer parameters.  
- **Reduced memory usage**.  
- **Teacher flexibility**: A teacher can transfer knowledge to many students.  

---
## Trade-Offs
- Requires access to the **teacher model** during training.  
- Student performance is limited: it cannot exceed the teacher (usually).  
- Training is more complex (needs soft logits and temperature tuning).  

---
## Practical Usage
- Widely used in **LLM families**:
  - DistilBERT (from BERT).  
  - TinyBERT, MiniLM.  
  - Distilled GPT variants.  
- Useful when deploying to **resource-constrained devices**.  
- Often combined with:
  - **Quantization** → for smaller precision.  
  - **Pruning** → for even smaller size.  

---
## Summary
- **Knowledge Distillation** transfers knowledge from a **large teacher model** to a **smaller student model**.  
- Uses a mix of **hard labels** and **soft teacher outputs** for training.  
- Produces models that are **smaller, faster, and more efficient**, with minimal accuracy loss.  
- A key technique for **LLM deployment and optimization**.  

---
## See Also
- [[Inference Optimization]]