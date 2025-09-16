# Anchors
#cv/object-detection/set-prediction/anchors #yolo/anchors 

An object detection method where predefined reference boxes of different sizes and aspect ratios are placed on a grid.
The model predicts offsets and class labels to adjust these anchors into final bounding boxes.

![Anchors](https://www.thinkautonomous.ai/blog/content/images/2023/05/Screenshot-2023-05-02-at-12.07.31.png)

## Algorithm
1. Predefine a set of anchor boxes at each grid cell of feature map
2. For each anchor, the model predicts
   - A class label
   - Offsets
3. Apply offset to anchors to form final bounding box predictions.
4. Apply [[Non-Maximum Suppression|NMS]]

## Related Models
`Faster R-CNN` (2015)
Introduced **Regional Proposal Network (RPN)** which used anchors at each grid cell to generate proposals.
Proposals are then classified & refined in the second stage.

`YOLOv2` (2016)
First YOLO to adopt anchors
Anchors improved its ability to detect objects of different aspect ratio.

`YOLOv3` & `YOLOv4`
Use multiple anchors per cell
Use feature pyramid for predictions at multiple scales

`RetinaNet` (2017)
Fully one-stage detector.
Anchors at multiple aspect ratio per feature map location.
Introduce **focal loss** to deal with anchor imbalance.

## See Also
- [[Region Proposals]]
- [[Window Centers]]
