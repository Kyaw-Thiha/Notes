# Radar Region Proposal Network
`RRPN` maps [[Radar|radar detection]] to camera coordinates and generating anchor boxes around these mapped points.

This achieved faster processing time than `Selective Search` algorithm of `Fast R-CNN`.

---
## Main Algorithm
The radar detections are mapped to camera coordinates to generate anchor boxes.
They are then used by a `two-stage` object detector.

But in order to map these detections properly, three things must be considered:
- `Perspective Transformation`
- `Anchor Generation`
- `Distance Compensation`

---
### Perspective Transformation
**Purpose**: Map radar detections from vehicle coordinates to camera image coordinates

The projection from 3D point to 2D image coordinate can described as
$$
p = HP
$$
where
- $p = [x,y,1]$ is the `2D camera coordinate`
- $P = [X, Y, Z, 1]$ is the `3D point` of radar detection
- $H$ is the `calibration matrix`
$$
H = \begin{bmatrix}
h_{11} & h_{12} & h_{13} & h_{14} \\
h_{21} & h_{22} & h_{23} & h_{24} \\
h_{31} & h_{32} & h_{33} & h_{34} \\
\end{bmatrix}
$$

---
### Anchor Generation

**Challenge**: 
- Radar POIs don't always map to `object centers`
- Radar doesn't provide `size information`

**Solution**: 
Generate multiple anchor boxes per POI with 
- $4$ `different sizes`, 
- $3$ different `aspect ratios` and 
- $4$ different `spatial alignments` $(\text{center, right, bottom, left})$.

---
### Distance Compensation
`Object appearance size` inversely relates to distance from camera

Hence, we use the range information of radar data to scale all generated anchors using
$$
S_{i} = \alpha \frac{1}{d_{i}} + \beta
$$
where
- $S_{i}$ is the `scaling factor`
- $d_{i}$ is the `distance` to the $i^{th}$ object
- $\alpha$ and $\beta$ are two `hyperparameters`

**Parameter Learning** 
$\alpha$ and $\beta$ are optimized by maximizing [[Intersection over Union (IoU)]] with ground truth boxes across training data:
$$
\arg\max_{\alpha, \beta}
\sum^N_{i=1} \sum^{M_{i}}_{j=1}
\max_{1 < k < A_{i}} 
\text{IoU}^i_{jk} (\alpha, \ \beta)
$$
These parameters are determined by a `grid search`.

---

