# Object Detection
#cv/tasks/object-detection
`Object Detection` detects the object type as well as its location in the image.

![Object Detection|400](https://areeblog.com/wp-content/uploads/2025/05/1446e76-f181-6047-4e73-8d8ba3c6a50e_object_detection_1.webp)

Formally given image $I$, object detection predicts a tuple
$$
\{ (b_{i}, c_{i}, s_{i}) \}
$$
where
- $b_{i}$ is the `bounding box`
- $c_{i}$ is the `class label`
- $s_{i}$ is the `confidence score`

---
## See Also
- [[Object Detection Models]]
- [[Computer Vision/Tasks/Object Detection/Metrics/Confusion Matrix|Confusion Matrix]]
- [[Intersection over Union (IoU)]]
- [[Mean Average Precision (mAP)]]