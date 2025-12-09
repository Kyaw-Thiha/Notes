# Interpreting Learning Curves: Common Patterns

> **Goal**: Learn how to read different shapes of learning curves (training vs validation) to diagnose training behavior in LLMs and other neural networks.

---

## 1. Ideal Learning Curve
![Ideal Learning Curve](learning_curve_ideal.png)

- **Training loss**: decreases smoothly.  
- **Validation loss**: decreases, then plateaus at a low value.  
- **Gap**: small difference between training and validation losses.  

✅ Indicates good generalization and stable training.  

---

## 2. Underfitting
![Underfitting Learning Curve](learning_curve_underfitting.png)

- **Training loss**: stays high, decreases very slowly or not at all.  
- **Validation loss**: remains high, similar to training loss.  

❌ Model fails to capture underlying patterns in the data.  

### Causes
- Model too small (not enough capacity).  
- Not trained long enough.  
- Learning rate too high → prevents convergence.  

### Fixes
- Larger model or more layers.  
- Train for more epochs.  
- Reduce learning rate.  
- Improve features / data quality.  

---

## 3. Overfitting
![Overfitting Learning Curve](learning_curve_overfitting.png)

- **Training loss**: continues to decrease steadily.  
- **Validation loss**: decreases initially, then rises after some point.  
- **Gap**: large divergence between training and validation.  

❌ Model memorizes training data but fails to generalize.  

### Causes
- Model too complex for available data.  
- Insufficient regularization.  
- Too few training samples.  

### Fixes
- Add regularization (dropout, weight decay).  
- Data augmentation or larger dataset.  
- Use early stopping (stop at validation loss minimum).  

---

## 4. Erratic Curves (Noisy Training)
### 4a. Erratic Validation Only
![Erratic Validation Learning Curve](learning_curve_erratic_validation.png)

### 4b. Both Erratic
![Erratic Learning Curve](learning_curve_erratic.png)

- **Loss curves** fluctuate heavily (zig-zag pattern).  
- Both training and validation losses are unstable.  

❌ Indicates unstable optimization.  

### Causes
- Learning rate too high.  
- Very small batch sizes → high variance in gradients.  
- Poor data preprocessing (e.g., wrong labels, inconsistent formatting).  

### Fixes
- Lower learning rate.  
- Increase batch size.  
- Check and clean dataset.  
- Use gradient clipping.  

---

## 5. High Variance with Plateaus
![High Variance with Plateau Learning Curve](learning_curve_high_variance.png)

- Training loss decreases, but **validation loss oscillates** or plateaus early.  
- Accuracy metrics may stagnate.  

⚠️ Model learns partially but struggles to generalize.  

### Fixes
- Try different learning rate schedules (warmup, cosine decay).  
- Collect more data.  
- Tune regularization strength.  

---

## 6. Diverging Loss
![Diverging Learning Curve](learning_curve_diverging.png)

- **Training loss** increases instead of decreasing.  
- Validation loss also increases or becomes NaN.  

❌ Model is not training at all.  

### Causes
- Learning rate way too high.  
- Numerical instability (exploding gradients).  
- Wrong model/dataset mismatch.  

### Fixes
- Reduce learning rate drastically.  
- Apply gradient clipping.  
- Double-check data pipeline and model architecture.  

---

## 7. Data Leakage (Suspiciously Perfect Curves)
![Data Leakage Learning Curve](learning_curve_leakage.png)
- **Training loss**: decreases smoothly to very low.  
- **Validation loss**: also decreases unusually well, sometimes even below training loss.  

⚠️ Suspicious: validation data may overlap with training data (data leakage).  

### Fixes
- Re-check dataset splits.  
- Ensure no duplicates across train/val/test.  

---

## 8. Summary Table

| Curve Pattern         | Training Loss | Validation Loss | Diagnosis       | Fixes |
|-----------------------|---------------|-----------------|----------------|-------|
| Ideal                 | ↓ smooth      | ↓ then stable   | Good training  | — |
| Underfitting          | High, flat    | High, flat      | Too simple     | Larger model, longer training |
| Overfitting           | ↓ steadily    | ↓ then ↑        | Memorization   | Regularization, early stopping |
| Erratic               | Fluctuating   | Fluctuating     | Unstable       | Lower LR, bigger batch |
| Diverging             | ↑ increasing  | ↑ increasing    | Broken training| Lower LR, gradient clipping |
| High Variance Plateau | ↓ some        | Oscillates/flat | Partial generalization | Tune LR, more data |
| Data Leakage          | ↓ smooth      | ↓ suspiciously  | Leakage issue  | Re-check data split |

---

## 9. Practical Tips

- Always plot **both training & validation** curves.  
- Don’t rely only on loss → track metrics like accuracy, F1, perplexity.  
- Use **moving averages** to smooth noisy curves.  
- Combine curves with **W&B** or **TensorBoard** for interactive monitoring.  

---
## See Also
- [[Learning Curves]]