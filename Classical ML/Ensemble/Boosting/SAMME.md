# SAMME 
#ml/ensemble/boosting/adaboost/SAMME

`SAMME` is the multi-class variant of [[AdaBoost]].

## Algorithm 

In `SAMME`, each base model $h_{k}(x)$ predicts a `class label` and the error weight is defined as
$$
\alpha_{k} = \ln\left( \frac{1 - \epsilon_{k}}{\epsilon_{k}} \right) + \ln(K-1)
$$
where $K$ is the number of classes in `Classification`

Then, the final model predicts by `majority vote`
$$
F(x) = argmax_{k} \sum_{t} \alpha_{k} \ I[h_{k}(x) = k]
$$
---
## See Also
- [[AdaBoost]]
- [[AdaBoost R2]]
