# Window Center
#cv/object-detection/set-prediction/window-center 

An object detection method where every pixel is treated as possible object center, and the model predicts the distances from that center to the four sides of the bounding box.
![Window Center](https://pub.mdpi-res.com/electronics/electronics-09-02075/article_deploy/html/images/electronics-09-02075-ag.png?1607213401)

## Algorithm
- Each pixel is treated as a candidate center of an object.
- At each center, the model predicts:
  - The object class
  - The distance to the four sides of the bounding boxes

## Related Models

`DenseBox` (2015)
- Trained on feature maps from a CNN
- Treated each pixel as possible object centers
- Directly predicted:
	- Class Score
	- Bounding box offsets

`UnitBox` (2016)
- Built on `DenseBox`
- Proposed a special **IoU loss function** to better train offset predictions
- Helped improve localization accuracy.

`CornerNet` (2018)
- Instead of predicting centers, it predicted corners (top-left, bottom-right) as keypoints
- Grouped centers to form boxes.

`FCOS (Fully Convolutional One-Stage Detector)` (2019)
- Directly predict distances to box edges
- Compared to `DenseBox`,
  - **Feature Pyramid Network (FPN)**: Handle object at multiple scales
  - **Center-ness Branch**: Predict how close a pixel is to true object center
  - **IoU-based loss** function

## Comparism to Sliding-Window
Although it sounds similar, it is fundamentally different to sliding window method.

`Sliding Window`: Pre deep-learning method where we explicitly crop a window, and classify if an object is in that crop.
`Window Center`: Use 1 CNN pass over an image, and treat every feature map pixel as possible center.

## See Also
- [[Anchors]]
- [[Region Proposals]]
