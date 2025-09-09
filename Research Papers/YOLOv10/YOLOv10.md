# YOLOv10: Real-Time End-to-End Object Detection
#research #cv/object-detection #yolo/v10

Link: https://arxiv.org/abs/2405.14458

Compared to YOLOv8,
- It replaces [[Non-Maximum Suppression|NMS]] with end-to-end object detection through **dual label assignment**.
- It improves efficiency by using **lightweight classification head**, **spatial channel decoupled downsampling** and **rank-guided block design**.
- It improves accuracy by using **larger convolution kernal**  and **partial self-attention**.

---
## Dual Label Assignment
- YOLOv10 use one-to-many & one-to-one heads during training, by aligning them through [[Dual Head Assignment#Prediction-Aware Score|predication aware score]].
- Then during inference, only the one-to-one head is used.
- This allows the model to no longer require [[Non-Maximum Suppression|NMS]] during inference.

[[Dual Head Assignment|Read More]]

## Improving Efficiency
- **Lightweight Classification Head**
  Regression head has more significance on performance, so classification head is made lightweight.
- **Spatial Channel Decoupled Downsampling**
  Decoupling spatial reduction & channel increase operations make the downsampling process more efficient.
- **Rank-Guided Block Design**
  Rank the blocks inside model based on performance, and keep replacing the lowest scoring blocks with 'cheaper' blocks till performance degradation is observed.