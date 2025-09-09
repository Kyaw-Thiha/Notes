# YOLOv10: Real-Time End-to-End Object Detection
#research #cv/object-detection #yolo/v10

Link: https://arxiv.org/abs/2405.14458

Compared to YOLOv8,
- It replaces [[Non-Maximum Suppression|NMS]] with end-to-end object detection through **dual label assignment**.
- It improves efficiency by using 
  - **lightweight classification head** 
  - **spatial channel decoupled downsampling**  
  - **rank-guided block design**.
- It improves accuracy by using 
  - **larger convolution kernal**   
  - **partial self-attention**.

---
## Dual Label Assignment
- YOLOv10 use one-to-many & one-to-one heads during training, by aligning them through [[Dual Label Assignment#Prediction-Aware Score|prediction aware score]].
- Then during inference, only the one-to-one head is used.
- This allows the model to no longer require [[Non-Maximum Suppression|NMS]] during inference.

[[Dual Label Assignment|Read More]]

## Improving Efficiency
- **Lightweight Classification Head**
  Regression head has more significance on performance, so classification head is made lightweight.
- **Spatial Channel Decoupled Downsampling**
  Decoupling spatial reduction & channel increase operations make the downsampling process more efficient.
- **Rank-Guided Block Design**
  Rank the stages inside model based on [[YOLOv10 Efficiency#Intrinsic Numerical Rank|Intrinsic Numerical Rank]], and keep replacing the lowest scoring blocks with [[YOLOv10 Efficiency#Compact Inverted Blocks (CIB)|CIB]] blocks till performance degradation is observed.

[[YOLOv10 Efficiency|Read More]]

## Improving Accuracy
- **Large Kernel Size**
  Use `7x7 depthwise conv` in [[YOLOv10 Efficiency#Compact Inverted Blocks (CIB)|CIB]] blocks for smaller models.
- **Partial Self-Attention**
  Partition features into two parts
  Send only 1 part through `self-attention + FFN`.
  Concatenate the 2 parts and fuse with `1x1 Conv`

[[YOLOv10 Accuracy|Read More]]


