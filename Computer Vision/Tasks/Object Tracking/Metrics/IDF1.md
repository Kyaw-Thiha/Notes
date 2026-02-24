# IDF1 (Identity F1-Score)
#cv/object-tracking/metrics/idf1

`IDF1` is a metric used for [[Object Tracking]], which calculates the `Bijective Matching` at track level.

$$
\text{IDF1} = \frac{|\text{IDTP}|}{|IDTP| + 0.5\ |IDFN| + 0.5\ |IDFP|}
$$
where
- $\text{IDTP}$ is the number of `true positive identity matches`
- $\text{IDFN}$ is the number of `false negative identity matches`
- $\text{IDFP}$ is the number of `false positive identity matches`

Note that compared to [[MOTA]], `IDF1` is biased towards association performance, and is sensitive to identity switches.

---
`ID-Recall`
$$
\text{ID-Recall} = \frac{|IDTP|}{|IDTP| + |IDFN|}
$$
`ID-Recall` measures the percentage of `true positive identity matches` relative to the number of ground truth matches.

---
`ID-Precision`
$$
\text{ID-Precision} = \frac{|IDTP|}{|IDTP| + |IDFP|}
$$
`ID-Precision` measures the percentage of `true positive identity matches` relative to the number of predicted matches.

---
`Relation to IDF1`

The `IDF1` metric combines both `IDP` and `IDR` into a single metric, providing a balanced evaluation of identity matching performance.

---
## See Also
- [[HOTA]]
- [[MOTA]]
- [[MOTP]]
