# Mean Average Precision (mAP)
#cv/object-detection/mAP 

`mAP (mean Average Precision)` is a metric used to measure the performance of `object detection` models.

![mAP](https://images.prismic.io/encord/1e3870ac-9b13-4749-b84a-e809733f6f5a_Mean+Average+Precision+in+Object+Detection-+Encord.png?auto=compress%2Cformat&fit=max)

---
`Recall & Precision`

`Precision` is number of correctly predicted positive predictions, given all positive detection by the model.
$$
\text{Precision} = \frac{TP}{TP + FP}
$$

`Recall` is number of correctly predicted positive predictions, given all the actual positive predictions. 
$$
\text{Recall} = \frac{TP}{TP + FN}
$$
[[Computer Vision/Object Detection/Metrics/Confusion Matrix|Read More]]

---
`IoU`
IoU measures how much two bounding boxes overlap relative to their total area.
$$
\text{IoU} 
= \frac{\hat{y} \ \cap \ y}{\hat{y} \ \cup \ y}
$$

[[Intersection over Union (IoU)|Read More]]

---
`AP`

First, we compute the no. of predictions whose [[Intersection over Union (IoU)|IoU]] $> \text{ threshold}$, where $threshold$ is usually $0.5$.

Second, we compute the `Precision` and `Recall`.
Then, using these results, we build the [[Precision-Recall Curve]].

Thirdly, we compute the area under the `Precision-Recall Curve`, and define it as `AP`.

---
`mAP`
To compute the `mean Average Precision`, we compute the mean of `AP` across all the classes.

---
## Types of mAP

`VOC mAP@0.50`
Build the [[Precision-Recall Curve]] with [[Intersection over Union (IoU)|IoU = 0.5]].
Then, there are 2 ways to find the area under the graph
- Sample at $11$ recall points $(0.0, 0.1, \dots, 1.0)$
- Modern version directly integrate over the entire area

`COCO mAP@0.50-0.95`
Build the [[Precision-Recall Curve]] with [[Intersection over Union (IoU)|IoU]] threshold at $10$ different points - $(0.50, 0.55, \dots, 0.90, 0.95)$.
For each of these points, compute the area under the curve.
Then, get the mean `AP` cross all the different threshold.
$$
AP_{\text{COCO}} 
= \frac{1}{10}(AP_{50} + AP_{55} + \dots + A_{95})
$$


![mAP|500](https://learnopencv.com/wp-content/uploads/2022/08/mean-average-precision-map-calculation-11-point-interpolation-pascal-voc-manual.gif)

---
## See Also
- [[Computer Vision/Object Detection/Metrics/Confusion Matrix|Confusion Matrix]]
- [[Intersection over Union (IoU)]]
- [[Precision-Recall Curve]]