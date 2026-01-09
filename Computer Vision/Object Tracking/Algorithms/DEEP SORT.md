# DEEP SORT
#cv/object-tracking/algorithm/deep-sort 
`Deep SORT` combine the motion and appearance information for more accurate associations.

![Deep SORT|500](https://www.researchgate.net/publication/358134782/figure/fig3/AS:11431281351731807@1743777218277/Overview-of-the-object-tracking-Deep-SORT-algorithm.tif)

It aims to fix [[SORT]]'s high frequency of identity switches due to low precision of motion model.

---
### Re-ID Model
The `Re-ID Model` is used for $\text{appearance feature extraction}$.

Specifically,
1. The $\text{image crop}$ is generated using the bounding box.
2. $a)$ A [[Convolutional Neural Network (CNN)|CNN]] is trained on $\text{re-identification dataset}$ to discriminate between object's identities.
   $b)$ The $\text{image crop}$ is fed into this [[Convolutional Neural Network (CNN)|CNN]] to extract each detected object's $\text{appearance embeddings vector}$.
   $c)$ The $\text{embeddings}$ are passed through $\text{softmax classfication head}$ to output object's identity.
3. During inference,
	- $\text{Classification Head}$ is dropped
	  $\text{(due to new identities not present in training dataset)}$
	- $\text{Cosine Distance}$ is used to compare feature embeddings
	- It can be denoted as
$$
d_{A} \ (i, j, E^D, E^T)
= \min_{E^T_{j,k} \in E^T_{j}}
\ d_{\cos} (E^D_{i}, \ E^T_{j,k})
$$
		where
		- $d_{\cos}$ is the [[Distance Measure|Cosine Distance]]
		- $i$ is the index of $i^{th}$ detection
		- $j$ is the index of $j^{th}$ track
		- $E^D$ represents the `detection appearance embeddings`
		- $E^T$ represents the `buffered track appearance embeddings`

---
## Cost Function
The `cost function` is defined as
$$
C_{\text{Deep SORT}} (i, j, \ D, T, \ E^D, E^T)
= \lambda \ d_{M}(D_{i}, \ \hat{T}_{j})
+ (1 - \lambda) \ d_{A}(i, j, \ E^D, E^T)
$$
where
- $D_{i}$ represents the $i^{th}$ `detection` with corresponding appearance embeddings $E_{i}$
- $\hat{T}_{j}$ represents the $j^{th}$ predicted `track bounding box` produced by the [[Kalman Filter]]
- $E_{i}$ is the embedding produced by the `ReID` for $i^{th}$ detection
- $E_{j} = \{ e_{j}^{(1)}, \ e_{j}^{(2)}, \ \dots, \ e_{j}^{(K)} \}$ is the set of past detection embeddings that we associated to the track $j$ 
- $d_{M}$ is the `Mahalanobis distance` 
- $d_{A}$ is the `appearance distance`
- $\lambda$ is the `weight` balancing motion and association based costs
	- If $\lambda=1$, the association cost is determined solely by motion & geometrics features.
	- If $\lambda=0$, the association cost is determined solely on appearance features.
	- `Deep SORT` uses $\lambda=0$ as default configuration.

Instead of relying on `negative IoU` like in [[SORT]], `Deep SORT` employs the `Mahalanobis Distance` which accounts for the uncertainty provided by the [[Kalman Filter]].

---
## Limitation of Mahalanobis Distance
- The predictions for lost tracks by the `Kalman Filter` accumulates uncertainty.
- This causes lost tracks to be cheaper to associate compared to active tracks.

`Cascaded Association`
To address the limitation of `Mahalanobis Distance`, `Deep SORT` uses cascaded assocation based on track's time lost.

[[Limitation of Mahalanobis Distance in Object Tracking|Read More]]

---
## Gate Function
The gate function is defined as
$$
\begin{align}
&G_{\text{Deep SORT}} (i, j, \ D, T, \ E^D, E^T)
\\[6pt]
&= \mathbb{1}[ 
\ d_{M}(D_{i}, \hat{T}_{j}) < t^{(M)} 
\ ]
\ \cdot \ 
\mathbb{1} [ \ 
d_{A}(i, j, \ E^D, E^T) < t^{(A)}
\ ]
\end{align}
$$
where
- $t^{(M)}$ is the `Mahalanobis distance threshold`
- $t^{(A)}$ is the `appearance distance threshold`

Association is possible only if both motion and appearance conditions are met.

---
## Other Improvements
`KF State`
For the [[Kalman Filter|KF]] state, it uses height and widths along with their velocity states.
By using it instead of scale and aspect ratio, it offers greater flexibility since height and width have decoupled velocities.
  
`Decoupled Assocation`
Detections are first associated with `active` tracks, before the `new` tracks.
This helps reduce the number of false positives.

---
## See Also
- [Main Reference Paper](https://arxiv.org/abs/2506.13457)
- [[Object Tracking]]
- [[SORT]]
- [[Limitation of Mahalanobis Distance in Object Tracking]]
- [[Kalman Filter]]
