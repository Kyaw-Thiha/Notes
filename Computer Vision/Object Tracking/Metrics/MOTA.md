# MOTA (Multi-Object Tracking Accuracy)
#cv/object-tracking/metrics/mota

`MOTA` is a metric used for [[Object Tracking]], which is good at evaluating the `association performance` (how well the objects are tracked).

$$
\text{MOTA} = 1 - \frac{|FP| + |FN| + |IDSW|}{gtDET}
$$
where
- $FP$ is total number of `false-positive detections`
- $FN$ is the total number of `false-negative detections`
- $IDSW$ is the total number of `identity switches`
- $gtDET$ is the total number of `ground-truth detections`

---

