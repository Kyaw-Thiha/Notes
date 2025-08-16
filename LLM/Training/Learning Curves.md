# Learning Curves 


> **Goal**: Understand what learning curves are, how to plot them, and how to interpret them for monitoring training progress of LLMs and other models.

---

## 1. What Are Learning Curves?

A **learning curve** is a plot that shows how a model’s performance changes over time during training.  

Typically, we plot:
- **Training loss** vs steps/epochs  
- **Validation loss/metric** vs steps/epochs  

These curves help us monitor whether the model is:
- Learning effectively  
- Overfitting to training data  
- Underfitting (not learning enough)  

---

## 2. Why Use Learning Curves?

- **Track progress**: See if the model is still improving.  
- **Detect overfitting/underfitting**: Compare training vs validation.  
- **Debug training issues**: Spot vanishing gradients, divergence, or poor hyperparameters.  
- **Guide hyperparameter tuning**: Adjust learning rate, batch size, or training duration.  

---

## 3. Example: Plotting Training vs Validation Loss

If using Hugging Face `Trainer`, metrics are automatically logged.  
If using a manual PyTorch loop, you can log values after each step/epoch.

```python
import matplotlib.pyplot as plt

train_losses = [0.9, 0.7, 0.5, 0.4, 0.35]
val_losses = [1.0, 0.8, 0.6, 0.55, 0.52]

plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Learning Curve")
plt.show()
```

---

## 4. How to Interpret Learning Curves

### Case A: Good Training
- Training loss decreases smoothly.  
- Validation loss decreases and stabilizes.  
- Small gap between training and validation curves.  
✅ Indicates model is learning generalizable patterns.

### Case B: Overfitting
- Training loss continues decreasing.  
- Validation loss bottoms out, then increases.  
❌ Indicates model is memorizing training data and losing generalization.  
→ Fix: regularization, dropout, early stopping, more data.

### Case C: Underfitting
- Both training and validation losses remain high.  
- Model fails to learn patterns from the data.  
→ Fix: larger model, longer training, better features, learning rate tuning.

See more curves at [[Interpreting Learning Curves]]

---

## 5. Metrics Beyond Loss

Loss is the most common, but learning curves can track other metrics:
- **Accuracy** (classification tasks)  
- **F1, Precision, Recall** (imbalanced datasets)  
- **Perplexity** (language modeling)  
- **BLEU/ROUGE** (translation, summarization)  

Plotting both **loss** and **metrics** provides a complete picture.

---

## 6. Practical Tips

- Always compare **training vs validation**.  
- Monitor **both short-term (per batch)** and **long-term (per epoch)** trends.  
- Use **smoothing** (e.g., moving average) for noisy curves.  
- Log curves with a tool like **Weights & Biases (W&B)** or **TensorBoard** for real-time monitoring.  
- Set up **early stopping** to automatically halt when validation stops improving.  

---

## 7. Summary

- **Learning curves** plot performance over time.  
- Key to understanding **whether your model is learning, overfitting, or underfitting**.  
- Compare **training vs validation curves** to spot issues.  
- Track **loss + task-specific metrics** for deeper insight.  
- Essential tool for **debugging and improving LLM fine-tuning**.  

---
## See Also
- [[Interpreting Learning Curves]]