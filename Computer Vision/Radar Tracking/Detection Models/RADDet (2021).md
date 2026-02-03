# RADDet 
### Range-Azimuth-Doppler based Radar Object Detection for Dynamic Road Users

This paper proposes three things:
- [[Radar]] dataset in `Range-Azimuth-Doppler` format
- Instance-wise `auto-annotation` method to build the dataset
- `One-stage` [[Computer Vision/Object Detection/Anchors|anchor-based]] detector that generates both
	- 3D bounding boxes on `RAD` domain
	- 2D bounding boxes on `Cartesian` domain
  
![RADDet|300](https://www.researchgate.net/publication/373388230/figure/fig6/AS:11431281212141949@1702544630746/RD-diagram-representation-of-the-RADDet-dataset-with-bounding-boxes-around-objects-and.png)

---
## Background Information
[[FMCW]] radar has been adopted in `Advanced Driver Assistance Systems (ADAS)`.

`Benefits` 
- Robust against adverse weather 
- Robust against lighting conditions

`Drawbacks`
- Low output resolution
- High sensor noise levels

### Data Processing
The raw `analog-to-digital (ADC)` signals from the radar are passed into `Digital Signal Processing (DSP)`.
This outputs either of
- Sparse `point cloud` representation
- Range-Azimuth-Doppler(`RAD`) spectrum

`RAD` spectrum keeps consistency of input data without any discretization.
But increases complexity, and runtime.

---
## Dataset Auto-Annotation

![RADDet Dataset Annotation](https://notes-media.kthiha.com/RADDet-(2021)/8674de3727b8f5b3e72ff470c3901269.png)

### Radar Processing

1. `Fast Fourier Transform` is performed on each dimensions of `ADC signals`.
   The output is the `RAD spectrum` of shape $(256, 256, 64)$.
2. [[Constant False Alarm Rate (CFAR)]] is employed to filter out noise signals on `RD Dimensions`.
3. The detections from `CFAR` is applied as mask.
4. `CFAR` suffer from poor detections when objects are at same range with similar speed.
   [[DBSCAN]] is used to remedy this.

### Stereo Image Labelling
Stereo vision is used for category labelling.
1. `Disparity maps` are generated from the rectified stereo image pairs using `Semi-Global Block Matching algorithm`.
2. Pretrained `Mask-RCNN` is applied on the left images to extract [[Instance Segmentation]] mask.
3. The prediction masks are then projected onto the `disparity maps`.
4. Using `triangulation`, the instance-wise point cloud outputs with predicted categories are generated.
5. Point cloud instances are transformed to the radar frame using a `projection matrix`.

---
## RadarResNet
### Data Processing
Raw Input: `RAD Tensor` of size $(256, \ 256, \ 64)$.

- For each value, `log` is applied.
- Then, `z-normalization` is applied
$$
I_{i_{norm}} = \frac{I_{i} - V_{\text{mean}}}{V_{\text{variance}}}
\quad , \ I_{i} \in D_{dataset}
$$

- `Range-Azimuth` axes is considered as input dimensions.
- `Doppler` axis is set as original channel size.

## Architecture
`Radar ResNet` uses a `ResNet` backbone with two detection heads.

![Radar ResNet](https://notes-media.kthiha.com/RADDet-(2021)/4a813d5d0e52ab9b770a8aadff67a11a.png)

### 3D Detection Head (RAD YOLO Head)
Output size of $3^{rd}$ dimension $(\text{Doppler axis})$ is set to $4$.

The `3D Detection Head` processes the feature maps into
$$
(16, \ 16, \ 4 \times \text{num of anchors} , \ (7 + \text{num of classes}) )
$$
where 
- $\text{First } 16$ is the height of bounding box
- $\text{Second } 16$ is the width of bounding box
- $4$ is the depth of bounding box
- $7$ consists of
	- objectness
	- 3D center point $[x, y, z]$
	- size $[w, h, d]$

The output is then fed into [[Non-Maximum Suppression (NMS)]].

### 2D Detection Head
Consists of two components: 
- `Coordinate Transformation Layer`
- `YOLO Head`

`Coordinate Transformation`
Traditionally, coordinate transformation from `range/azimuth` domain $[r, \theta]$ to `cartesian` width/depth domain $[x,z]$ is formulated as
$$
\begin{align}
x &= r.\cos(\theta) \\[6pt]
z &= r.\sin(\theta) \\[6pt]
\theta &\in \left[ -\frac{\pi}{2}, \frac{\pi}{2} \right]
\end{align}
$$

---
`Coordinate Transformation Layer`
Input feature maps are in shape $(16, 16, 256)$.
Hence, they can be interpreted as $256$ `RA features` of form $[r, \theta]$.

To replicate the coordinate transformation,
- Each `RA` feature maps is fed individually into two [[Neural Network|Fully Connected Layer]] for non-linear transformation.
- Output is reshaped to $(32, 16)$ by the layers.
- Concatenated to build Cartesian feature map $(32, 16, 256)$.
- One `ResNet Block` is used to post process this feature.
- Transformed outputs are fed into the `YOLO Head`.

---
## See Also
- [RADDet (2021)](https://arxiv.org/abs/2105.00363)