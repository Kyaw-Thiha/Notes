# Object Tracking Evaluation
#cv/object-tracking/metrics 

These are ways to evaluate the performance of [[Object Tracking]] models.

![Object Tracking](https://cdn.prod.website-files.com/67d7ebcca3cabb623904dcde/67d7f3194f3a9b303c8bef20_63dcc1770eebde67243a8cb0_palace-demo.gif)

---
`Error Types`
The different `association errors` in object tracking are
- $\text{False Positive Track}$: A tracked object does not exist.
- $\text{False Negative Track}$: Object not recognised throughout the video
- $\text{Identity Switch}$: Tracks have mixed detection matches.

The different `detection errors` in object tracking are
- $\text{False Positive Detection}$: Detecting non-existent object.
- $\text{False Negative Detection}$: Not detecting an existing object.
- $\text{Localization Error}$: Object's detection offsets from ground truth

Note that $\text{Localization Error}$ are not explicitly visualized as predicted detections almost never perfectly match the ground truth.

---
## Measures
To evaluation these errors, we need to consider 2 different measures.

`Similarity Measure` 
Measure of how similar predictions are to ground truth
  
`Bijective Matching`
All possible matching pairs form a `bipartite graph`.
Score threshold is applied to determine if two nodes can match.
It is commonly solved using [[Hungarian Algorithm]].
[[Bipartite Matching|Read More]]

---
## Evaluation Metrics
The evaluation metrics are
- [[MOTA]] is an earlier metrics that measure the [[Computer Vision/Object Detection/Metrics/Confusion Matrix|accuracy]].
- [[MOTP]] is an earlier metrics that measure the [[Computer Vision/Object Detection/Metrics/Confusion Matrix|precision]].
- [[IDF1]] measures the [[Computer Vision/Object Detection/Metrics/Confusion Matrix|F1 Score]] at a tracking level
- [[HOTA]] jointly consider detection, association and localizaiton.

---
## See Also
- [[MOTA]]
- [[MOTP]]
- [[IDF1]]
- [[HOTA]]
