# Scale-NMS
`Scale-NMS` is extension of [[Non-Maximum Suppression (NMS)|NMS algorithm]] for `BEV space`.

![Scale-NMS|500](https://notes-media.kthiha.com/Scale-NMS/5523d9ab03573f5bae5587b633f3a2a9.png)

---
## Algorithm
1. **Scale up** object bounding boxes according to `category-specific factors` before applying NMS
2. **Apply classical NMS** with the enlarged boxes 
   (now redundant detections will have non-zero IOU)
3. **Rescale back** to original sizes after suppression

---
## Background Information
[[Non-Maximum Suppression (NMS)|NMS]] relies on [[Intersection over Union (IoU)|IoU]] to identify and suppress redundant detections.
However, in BEV space, different object categories have vastly different physical footprints on the ground plane.

### Specific problem
For objects like `pedestrians` and `traffic cones`, their ground plane footprint is extremely small.

The `false positive predictions` of these objects may have zero overlap with the `true positive detection`.

This means $\text{IoU} =0$, so `NMS` can't identify them as redundant and fails to suppress them.

---
## Read More
- [[BEVDet (2022)]]
- [[Non-Maximum Suppression (NMS)]]
- [Original Paper](https://arxiv.org/abs/2112.11790)