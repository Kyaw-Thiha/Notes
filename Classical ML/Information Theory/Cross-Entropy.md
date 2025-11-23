# Cross-Entropy
#ml/information-theory/cross-entropy  

`Cross-Entropy` measures how well one probability distribution $P(x)$ predicts another probability distribution $Q(x)$.
$$
H(P, Q) = -\sum_{i=1}^N P(x_{i}) \log(Q(x_{i}))
$$
![Cross-Entropy](https://cdn.prod.website-files.com/62cd5ce03261cb3e98188470/6891d70bd0f8b0a5e3fc1448_AD_4nXfmkRKiEpMf82QxYZRg4L37LZOQUZcYJzbremo7VkytQIwYDmQvuHd-6AgEoQCdJpgLPdyz219F1hSETFEtXk_IR5jWFVcy1udSIeFyaCnr7ibYNX3qWV_s2Z6rvmx_v6ww2vCE.png)

---
From [[KL Divergence]], we can derive that 
$$
H(P,Q) = H(Q) + D_{KL}(P \ || \ Q)
$$
This means that `Cross-Entropy` encodes both the divergence between 2 distributions ($P$ and $Q$), as well as intrinsic unpredictability of the true distribution $Q$.

This means that minimizing `Cross-Entropy` will optimize our predictions to be closer to the ground truth.

---
## Cross-Entropy Loss
We can define `Cross-Entropy Loss` as
$$
L_{CrossEntropy} = -\sum_{i=1}^N y_{i} \log(\hat{y}_{i})
$$
where
- $\{ y_{i} \}_{i=1}^N$ is the true class label.
- $\{ \hat{y}_{i} \}_{i=1}^N$ is the predicted probability distribution.

![Cross-Entropy Loss](https://machinelearningknowledge.ai/wp-content/uploads/2019/07/Binary_Cross_Entropy.gif)

`Cross-Entropy` loss can usually be used as loss function for classification tasks.

Note that it heavily penalises extreme misclassification errors.
$$
\begin{align}
L_{i}(w) &= - [y_{i} \log(P_{i}) + (1 - y_{i}) \ \log(1-P_{i})] \\[6pt]

&= \begin{cases}
-\log(P_{i}) & y_{i} = 1 \\[6pt]
-\log(1 - P_{i}) & y_{i} = 0
\end{cases}
\end{align}
$$

---
## See More
- [[KL Divergence]]
- [[Entropy]]
- [[Learning Logistic Regression]]
