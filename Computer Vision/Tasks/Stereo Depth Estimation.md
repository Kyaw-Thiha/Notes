# Stereo Depth Estimation
`Stereo Depth estimation` is estimating the depth given two or more cameras.

![Stereo Depth Estimation|400](https://media.geeksforgeeks.org/wp-content/uploads/20250730153511255742/Stereo-Vision.webp)

---
## Problem Formulation
Note that unlike [[Monocular Depth Estimation]], `stereo camera depth estimation` has a closed-form geometric solution
$$
Z = \frac{f \cdot b}{d}
$$

where
- $f$ is the `focal length`
- $d$ is the `disparity` 
  (pixel shift between matched points in left/right images)

----
## Epipolar Constraint
For a rectified stereo pair, the corresponding points lie on the same horizontal scanline.

![Epipolar Geometry|300](https://ars.els-cdn.com/content/image/3-s2.0-B9780081004128000048-f04-03-9780081004128.jpg)

This collapse the 2D search problem into 1D search along each row.

---