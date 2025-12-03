## Dual Label Assignment
#research #yolo/v10 
In previous YOLO models, they are trained on one-to-many head (one ground truth to many bounding boxes), and during inference, post-processed using [[Non-Maximum Suppression (NMS)|NMS]].

One-to-many head helps improve accuracy.
But the problem is the increased time during inference.

To solve that, YOLOv8 used 2 heads during training: one-to-one and one-to-many head.

It then ensure the 2 heads are making same predictions  by training on [[#Prediction-Aware Score]].

Then during inference, only the one-to-one head is used, essentially allowing the model to bypass [[Non-Maximum Suppression (NMS)|NMS]] post-processing, while also not requiring to use Hungarian matching.

## Prediction-Aware Score
$$
m(\alpha, \beta) = s.p^a.IoU(\hat{b},b)^\beta
$$
where
- $p$ is the class score
- $IoU$ is the box overlap
- $s$ is the spatial prior {0, 1} (Decides whether to learn based on ground truth)
- $\alpha, \beta$ is the trade-off for classification vs localization
High $\alpha$ favours high-$p$ (confident) candidates, while high $\beta$ favours high $IoU$ candidates.
Default: $\alpha = 0.5, \beta=6$

### How it help match the 2 heads
Each head rank candidates on their own score.
- One-to-many: $m_{o_{2}m} = s.p^{\alpha_{o_{2}m}}.IoU^{o_{2}m}$
- One-to-one: $m_{o_{2}o} = s.p^{\alpha_{o_{2}o}}.IoU^{o_{2}o}$


Suppose $\alpha$ and $\beta$ of $o_{2}o$ and $o_{2}m$ are factors to each other.
- $\alpha_{o_{2}o} = r.\alpha_{o_{2}m}$
- $\beta_{o_{2}o} = r.\beta_{o_{2}m}$

Then, 
$$
\begin{align}
&m_{o_{2}o} \\
&= s.p^{r.\alpha_{o_{2}m}}.IoU^{r.\beta_{o_{2}m}} \\
&= (s.p^{\alpha_{o_{2}m}}.IoU^{\beta_{o_{2}m}}) ^ r\\
&= (m_{o_{2}m})^r
\end{align}
$$

Since $x\to x^r$ is strictly increasing for $x>0$,
$arg \ max \ m_{o_{2}o, j} = arg \ max \ m_{o_{2}m, j}$, the ranking is preserved.

### Metric
To measure the difference between the one-to-one & one-to-many head's scores, the paper used the [Wasserstein Metric](https://en.wikipedia.org/wiki/Wasserstein_metric).

### Prediction-Aware Score vs Loss
The loss used is binary cross-entropy (BCE).
So, how it works is the prediction-aware score decides which ground-truth region to learn (top-1 for one-to-one and top-n for one-to-many).
Then, the same loss function (BCE) is used for both heads to learn those region(s).

## See Also
- [[YOLOv10]]
- [[YOLOv10 Efficiency]]
- [[YOLOv10 Accuracy]]
- [[Non-Maximum Suppression (NMS)]]