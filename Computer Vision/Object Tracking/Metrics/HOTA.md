# HOTA (Higher Order Tracking Accuracy)
#cv/object-tracking/metrics/hota

`HOTA` is a metric used for [[Object Tracking]], which jointly consider the performance of `detection`, `association`, and `localization quality`.

$$
\text{HOTA}_{\alpha} = \sqrt{ DetA_{\alpha} \cdot AssA_{\alpha} }
$$
where 
- $DetA_{\alpha}$ is the `Jaccard Index`([[Intersection over Union (IoU)|IoU]])
$$
DetA_{\alpha} = \frac{|TP|}{|TP| + |FN| + |FP|}
$$
- $AssA_{\alpha}$ measures the `association performance`
$$
AssA_{\alpha} 
= \frac{1}{|TP|} \ \sum_{c \in \{ TP \}} A(c)
$$
where
$$
A(c) = \frac{|\text{TPA}(c)|}
{|\text{TPA}(c)| + |\text{FNA}(c)| + |\text{FPA}(c)|}
$$
and $c$ is a pair of ground truth and prediction IDs.

Specifically,
- $\text{TPA}(c)$ is the set of `true positives` ($TP \ s$) that have same `ground truth` and `prediction IDs`
- $\text{FNA}(c)$ is the union of 
	- Set of all $TP\ s$ that have same ground truth ID as $c$, but different prediction IDs
	- Set of all `unmatched ground truth` with same ground truth ID as $c$
- $\text{FPA}(c)$ is the union of 
	- Set of $TP \ s$ that have same prediction IDs as $c$, but a different ground-truth ID 
	- Set of all `unmatched ground truth` with same ground truth ID as $c$


---
`Final Calculation`

$\text{HOTA}_{\alpha}$ is calculated with different $19$ thresholds $(0.05 \to 0.95)$.
$$
\text{HOTA} = \frac{1}{19} \sum_{\alpha} 
\ \text{HOTA}_{\alpha}
$$
---
## See Also
- [[MOTA]]
- [[MOTP]]
- [[IDF1]]
