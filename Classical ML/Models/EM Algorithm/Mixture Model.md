# Mixture Model
#ml/classic-models/mixture-model 

`Probabilistic Mixture Model` assigns a point with a probability that the cluster generates it.

![Probabilistic Mixture Model|400](https://akireeva.com/assets/GMM/density_estimation.gif)

---
`Motivation`

Note that the follow `data distributions` are not optimal for [[K-Means]]
- Points near the boundary of several clusters
- Points from elongated data distribution

This is because `K-Means` uses a hard assignment.
To solve this, we can use `Probabilistic Mixture Model` which uses a soft assignment.

---
`Multimodal Distribution`
Data distribution can have multiple modes.
Using a single mode data distribution will incorrectly assign high probability to regions with very small amount of data.

![Multimodal Distribution|400](https://mdpi-res.com/electronics/electronics-12-01410/article_deploy/html/images/electronics-12-01410-g001-550.jpg)

---
`Problem Formulation`

The `Mixture Model` represents the probability distribution as a weighted sum of multiple data distributions

$$
p(y) = \sum^K_{j=1} m_{j} \ p(y; \theta_{j})
$$
$$
\begin{cases}
m_{j} \in [0, 1] \\[6pt]
\sum^K_{j=1} m_{j} = 1
\end{cases}
$$
where
- $m_{j}$ is the `weight`
- $p(y; \ \theta_{j})$ is the `component density function`
- $\theta_{j}$ is the `parameters` of $j^{th}$ density function

---


