# Anchors
#cv/object-detection/anchors 

`Anchors` are predefined `bounding boxes` with fixed aspect ratio placed densely across the image, at different scales.

![Anchors](https://fritz.ai/wp-content/uploads/2023/09/1CYTDLg54ol-NpBOnrhFo2A.jpeg)

For each `anchor`, the model predicts the `classification score`, and the `regression offset` $(\Delta x, \Delta y, \Delta w, \Delta h)$.

---

`Why Anchors Used to exist`

[[Convolutional Neural Network (CNN)|CNN]] feature maps have fixed `receptive field`, so they can't represent arbitrary box sizes.
Hence, `anchors` act as priors to stabilize the training, and reduce the search space.

`Box Regression`

Given a anchor box of $(x, y , w, h)$, the model predicts $(t_{x}, t_{y}, t_{h}, t_{w})$ where
$$
\begin{align}
&t_{x} = \frac{x - x_{pred}}{w_{pred}} \\[6pt]
&t_{y} = \frac{y - y_{pred}}{h_{pred}} \\[6pt]
&t_{w} = \log\left( \frac{w}{w_{pred}} \right) \\[6pt]
&t_{h} = \log\left( \frac{h}{h_{pred}} \right) \\[6pt]
\end{align}
$$

---
`Problems with Anchors`

- We need hand-tuning for anchor sizes, since small objects have bad detection unless many anchors exist.
- However, more anchors implies more compute.
- Likewise, there is also class imbalance since most of the `anchors` ended up being background detection.

`Anchor-Free`
To alleviate these issues, different models use different strategies including
- Keypoint/Center based (`CenterPoint`)
- Per-Pixel Regression (`FCOS`)
- Transformer Queries ([[DETR]])
  These days, most models including `YOLO` and `DINO`, use this transformer approach.

---
## See Also
- [[Non-Maximum Suppression (NMS)]]
- [[Mean Average Precision (mAP)]]
