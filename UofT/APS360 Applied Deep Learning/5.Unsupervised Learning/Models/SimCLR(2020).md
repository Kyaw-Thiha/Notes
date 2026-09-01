# SimCLR 
[[SimCLR(2020)|SimCLR]] [[AutoEncoders Pretraining|pretrains]] an [[Convolutional Neural Network (CNN)|encoder]] via NT-XENT [[Contrastive Learning|contrastive loss]] on [[Data Augmentation|augmented image pairs]].

![200](https://miro.medium.com/1*aLVElJMw7JW5GeB0jWANlg.png)
where
- $\tau$ is a family of [[Data Augmentation|augmentation]].
- $f(\cdot)$ represent the encoder.
- $h_{i}$ is the latent representation.
- $g(\cdot)$ represent the prediction head. 
- $f(\cdot)$ and $h_{i}$ would then be used in other downstream tasks.

> It learns to maximize agreement between two views of same image relative to other in-batch samples.

---
## Contrastive Loss
**Setup**:
- For a batch $N$ images, [[SimCLR(2020)|SimCLR]] creates $2N$ augmented views.
- For each image, its $2$ augmented views form a positive pair.
- The other $2(N-1)$ views in the batch are treated as negative relative to that pair.

**Formula**
For a positive pair $(i,j)$,
$$
\ell_{ij}
= - \log \frac{\exp\left( \frac{\text{sim}(z_{i}, z_{j})}{\tau} \right)}
{\sum^{2N}_{k=1} \mathbb{1}_{k\neq i} \exp\left( \frac{\text{sim}(z_{i}, z_{j})}{\tau} \right)}
$$
where
- $z_{i}, \ z_{j}$: Projection-head outputs.
- $\text{sim(u,v)}$: [[Distance Measure|cosine similarity]] $\frac{u^{T}v}{||u|| \ || v ||}$
- $\tau$: temperature hyperparam. Scales similarity before softmax.
- Denominator: sums the similarity of $z_{i}$ against every other embedding in the batch (all negatives + the one positive), excluding itself

---
## See Also
- [[Contrastive Learning]]
- [[Distance Measure]]
- [[AutoEncoders Pretraining]]
- [[Transfer Learning]]
- [[Data Augmentation]]