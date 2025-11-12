# Ensemble Model
#ml/ensemble

`Ensemble Methods` combine the strengths of multiple simpler base models to build a more powerful model.

![Ensemble Methods](https://miro.medium.com/v2/1*OZPOQUKiaVmZOEMm_-8iYA.png)

It has 
- Higher accuracy
- Reduced overfitting risk
- Increased robustness to outliers

---
## Techniques
### [[Bagging]]

`Bootstrapping` sample of the dataset, and training $k$ number of base models on it, before `aggregating` all their predictions.

### [[Boosting]]
Sequentially training $k$ number of models, whereby each model learn from the error of the previous model.

### [[Stacking]]
Train multiple different `base models`, and then train a `meta-learner model` to combine their predictions optimally.

---
