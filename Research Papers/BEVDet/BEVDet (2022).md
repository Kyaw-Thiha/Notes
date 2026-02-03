# BEVDet
`BEVDet` is a camera-based 3D object detection framework for autonomous driving that operates in `Bird's-Eye-View (BEV) space`.

![BEVDeT|400](https://notes-media.kthiha.com/BEVDET/8167f6dd5762f49a9327823c48583afd.png)

---
## Architecture
`BEVDet` follows a similar architecture as used in `BEV Semantic Segmentation models`.
It consists of
- `Image View Encoder`: Encoding features in image view
- `View Transformer`: Transforming features from image to BEV
- `BEV Encoder`: Further encoding features in BEV
- `Head`: Task specific 

### Image-View Encoder
Encodes input features into high level features.

- `Backbone`: [[ResNet]] and [[SwimTransformer]] for high-level feature extraction
- `Neck`: [[Lift, Splat, Shoot|FPN-LSS]] for multi-resolution feature fusion

### View Transformer
Transforms the features from image into BEV.

- Takes `image-view feature` as input.
- Densely predicts the `depth` through a classification manner.
- Classification scores and the derived image-view feature are used in rendering the predefined `point cloud`.
- Apply a [[Pooling|pooling operation]] along the vertical direction to generate BEV Features.

### BEV Encoder
Further encodes the feature in the BEV space.

Compared to [[#Image-View Encoder]], it perceives some pivotal cues with high precision like scale, orientation, and velocity.

Similar to [[Lift, Splat, Shoot]], it uses
- [[ResNet]] to construct the backbone
- [[Lift, Splat, Shoot|FPN-LSS]] to combine the features with different resolution

### 3D Object Detection Head
Adopts the 3D object detection head in the first stage of [[CenterPoint]].

---
## Data Augmentation
The overfitting issue was observed due to excessive fitting capacity of `BEVDet` in the BEV space.

To tackle this issue, a data augmentation strategy was applied in the image view.

### Background Info
Recall that [[#View Transformer]] transforms feature from image to BEV in a pixel-wise manner.

This can be formulated as 
$$
p_{\text{camera}} = I^{-1} (p_{image} * d)
$$
where
- $d$ is a `depth` value
- $p_{\text{image}} = [x_{i}, y_{i}, 1]^T$ is a pixel on the `image`

Common data augmentation strategies can be formulated as a $3 \times 3$ transformation matrix $A$.
When a data augmentation strategy is applied the `input image`,
$$
p'_{\text{image}} = A p_{\text{image}}
$$

an inverse transformation $A^{-1}$ needs to be applied in the `view transformation`
$$
p'_{\text{camera}}
= I^{-1} (A^{-1} \ p'_{image} * d)
= p_{\text{camera}}
$$

### Data Augmentation on BEV
As the [[#View Transformer]] isolates the two view spaces in augmentation perspective, another augementation strategy specific on regularizing learning in `BEV` space is implemented.

---
## Scale-NMS
An improved [[Non-Maximum Suppression (NMS)|NMS]] algorithm is used to suppress small objects in the `BEV space`.

[[Scale-NMS|Read More]]

---
## See More
- [Paper](https://arxiv.org/abs/2112.11790)

