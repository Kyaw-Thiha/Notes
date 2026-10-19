# PointNet++
#research #cv/3d/object-detection/point-net-plusplus #cv/3d/object-segmentation/point-net-plusplus

Paper: https://arxiv.org/abs/1706.02413

[[PointNet]] was the first model to be able to capture features from `3D Point Clouds`, and use them for object detection and semantic segmentations.

However, it fails to capture the `local structures` effectively.

![PointNet++](https://miro.medium.com/v2/resize:fit:1200/1*FZgxecHJkI9pQcVPL4hs0Q.png)

`PointNet++` propose a `hierarchical neural network` of [[PointNet]] layers.

This is meant act as a [[Convolution Layer|CNN]], but 
- `CNN` capture local features in early layers, and global features in later layers.
- `Hierarchical Neural Network` capture global features in early layers, and local features in later layers.

The key is to partition the set of points into overlapping local regions

