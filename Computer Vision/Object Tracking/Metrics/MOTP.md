# MOTP (Multi-Object Tracking Precision)
#cv/object-tracking/metrics/motp

`MOTP` is a metric used for [[Object Tracking]], which is good at evaluating the `localization performance` (how well the bounding boxes are).

$$
\text{MOTP} = \frac{1}{|TP|} \sum_{m\in TP} S_{m}
$$
where
- $TP$ is the total number of `true positives`
- $S_{m}$ is the `similarity score` of the matched pair $m$ ([[Intersection over Union (IoU)|IoU]] or other [[Distance Measure]])

---


