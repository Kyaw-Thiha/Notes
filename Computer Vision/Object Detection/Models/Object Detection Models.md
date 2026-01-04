# Object Detection Models
#cv/object-detection/models 


![Object Detection|400](https://areeblog.com/wp-content/uploads/2025/05/1446e76-f181-6047-4e73-8d8ba3c6a50e_object_detection_1.webp)

---
### Proposal-Based Object Detectors
`Proposal-based` object detectors follow a two-stage pipeline:
1. First, region proposals are generated
2. Second, each proposed region undergoes bounding box localization and classification.

Note that the inference performance of these methods depends on hyperparameter of [[Non-Maximum Suppression (NMS)]].

![Propoal-Based Object Detector|400](https://b2633864.smushcdn.com/2633864/wp-content/uploads/2020/06/region_proposal_object_detection_output_beagle_before.png?lossy=2&strip=1&webp=1)

`R-CNN (2014)`
R-CNN was one of the earliest method.
- Generates region proposals using selective search algorithm
- Then, applies a [[Convolutional Neural Network (CNN)]] classifier for each proposal

Although it achieves high accuracy at its time, it has slow computation speed.

`Fast R-CNN (2015)`
Its main algorithm was
1. Uses single [[Convolutional Neural Network (CNN)|CNN]] pass to obtain image features
2. Introduces `Region of Interest (ROI)` [[Pooling]].
   This pools image features for each proposal obtained through selective search algorithm.

Its main innovations were
- Sharing convolutional features across proposals.
  Helps improves efficiency.
- Introducing [[Computer Vision/Object Detection/Anchors|Anchors]].
  These are predefined bounding boxes of various sizes and aspect ratios to serve as reference point for object localization.
  Helps improves accuracy.

`Faster R-CNN (2015)`
- This replaces selective search with `region-proposal networks (RPN)`.
- Enables end-to-end training and significantly improves speed.

`Mask R-CNN (2017)`
Added a branch for [[Instance Segmentation|Pixel-Wise Instance Segmentation]].

---
### Grid-Based Object Detectors
`Grid-Based` object detectors 
- divides the image into a grid where each grid cell is responsible for detecting one or more objects
- formulates object detection into a `regression` problem
- uses single-shot design that directly predicts bounding boxes and class probabilities from an image.

![Grid-Based Object Detector|400](https://media.licdn.com/dms/image/v2/D5622AQFR9XHhWHAgag/feedshare-shrink_800/B56ZkX65yNHMAs-/0/1757042950667?e=2147483647&v=beta&t=4fpGVayzUMGbudf-nhGEFkYbwPuklvbslopOW_lKdlY)

`YOLO (2016)`
The grid-based approach allow `YOLO` to be state-of-the-art in real-time performance.
- Uses the `YOLO loss` for its regression problem.

$$
\mathcal{L} 
= \lambda_{coord} \ \mathcal{L}_{bbox}
+ \mathcal{L}_{obj}
+ \lambda_{noobj} \ \mathcal{L}_{noobj}
+ \mathcal{L}_{class}
$$

- Main weakness is detecting small objects due to design of its `regression loss`

`YOLOv2 (2017)`
- Introduces bag of tricks to improve accuracy $(\text{Small Object Detection})$ and speed
- Introduces [[Computer Vision/Object Detection/Anchors|Anchors]] to improve localization accuracy

`YOLOv3 (2018)`
Enhances multi-scale predictions using [[Feature Pyramid Network (FPN)]], further improving robustness for small object detection.

`YOLOv4 (2020)`
Introduces an extended bag of tricks including
- `CSP DarkNet` as a backbone
- `Mosaic Augmentation`

---
### Query-Based Object Detectors
`Query-based` object detectors reformulates object detection as a set prediction problem.

It consists of four main components:
- [[Convolutional Neural Network (CNN)|CNN]] backbone $(\text{typically ResNet})$ to extract image features
- [[Encoder|Transformer Encoder]] to enrich image features using [[Self-Attention]]
- [[Decoder|Transformer Decoder]] to predict object embeddings based on learnable queries $(\text{learnable positional encodings})$ through `Cross-Attention` with image features
- [[Neural Network|Prediction Head]] that maps $\text{object embeddings}$ to $\text{bounding box coordinates}$ and $\text{class labels}$.

![Query-Based Object Detector|500](https://moonlight-paper-snapshot.s3.ap-northeast-2.amazonaws.com/arxiv/dynamic-object-queries-for-transformer-based-incremental-object-detection-1.png)

`DETR (2020)`
`Detection Transformer` eliminates the need for [[Computer Vision/Object Detection/Anchors|Hand-Crafted Anchors]] and post-processing steps like [[Non-Maximum Suppression (NMS)|NMS]].
- Training process using [[Bipartite Matching]] to assign ground-truth objects to predictions.
- [[Hungarian Algorithm]] ensures unique `one-to-one` assignment.
- Lack of duplicate predictions avoid the need of [[Non-Maximum Suppression (NMS)|NMS]].
- This enables `end-to-end` training and inference.

However, it also had drawbacks like
- Slow convergence, especially for small objects
- Slower inference time compared to `YOLO models`

[[DETR|Read More]]

`DAB-DETR (2022)`
Improves upon [[DETR]] with 
- Introduces dynamic anchor boxes
- Address convergence speed
- Improve small object detection issues

`DINO (2022)`
Improves upon [[DETR]] accuracy and robustness with 
- Advanced query initialization
- Contrastive Denoising

`RT-DETR (2023)`
Real-time variant of [[DETR]].

---
## See Also 
- [Main Reference Paper](https://arxiv.org/abs/2506.13457)