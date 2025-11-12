# Boosting
#ml/ensemble/boosting

`Boosting` is an [[Ensemble Model]] that builds models sequentially, with each one learning from the mistakes of the previous model.

![Boosting](https://substackcdn.com/image/fetch/$s_!Gt1_!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff66c6d20-d1ca-4f56-be77-07e12a3d1981_786x1098.gif)

It is good at `Bias Reduction`, since keeps refining decision boundary to fit complex data.

---
## Technique

First, train the model on the original data.
Second, we compute error on each data $x_{i}$.
Thirdly, we reweigh each training sample to focus on those errors.
Fourth, we fit new model on this weighted dataset.

The final `prediction` is a weighted sum (or vote) of all the models, with later models having more weight.

## Variants
- [[AdaBoost]]
- [[Gradient Boosting]]

---
## Limitations
- `Sequential` $\implies$ slower training
- Sensitive to noise & outliers
- High overfitting risk

---
## See Also
- [[Ensemble Model]]
- [[Bagging]]
- [[AdaBoost]]
- [[Gradient Boosting]]
