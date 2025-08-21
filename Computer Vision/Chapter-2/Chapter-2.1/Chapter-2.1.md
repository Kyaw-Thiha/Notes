#cv/transformations  
## (2.1.1) 2D Transformations

**Non-Homogeneous**: 
- Normal Mathematical Representation.
**Partially Homogeneous**: 
- Using 2x3 Affine Matrix to transform. 
- At the end of transformation, 2D vector is produced.
**Homogeneous**: 
- Using 3x3 Matrix to transform. 
- At the end of transformation, augmented 3D vector is produced.
- Used in robotics & computer vision.

| Transformation    | Matrix                                        | No. of Degree of Freedom | Preserves      |
| ----------------- | --------------------------------------------- | ------------------------ | -------------- |
| Translation       | $\begin{bmatrix}I&t\end{bmatrix}_{2x3}$       | 2                        | Orientation    |
| Rigid (Euclidean) | <br>$\begin{bmatrix}R&t\end{bmatrix}_{2x3}$   | 3                        | Length         |
| Similarity        | <br>$\begin{bmatrix}s.R&t\end{bmatrix}_{2x3}$ | 4                        | Angles         |
| Affine            | $\begin{bmatrix}A\end{bmatrix}_{2x3}$         | 6                        | Parallelism    |
| Projective        | $\begin{bmatrix}\tilde{H}\end{bmatrix}_{3x3}$ | 8                        | Straight Lines |

### Main Transformations
[[1. 2D-Translation]]
[[2. 2D-Rotation]]
[[3. 2D-Scaled Rotation]]
[[4. 2D-Affine]]
[[5. 2D-Projective]]
### Additional Transformations
[[6. 2D-Stretching]]
[[7. 2D-Planar Surface Flow]]
[[8. 2D-Bilinear Interpolant]]


## (2.1.2) 3D-Transformation
| Transformation    | Matrix                                        | No. of Degree of Freedom | Preserves      |
| ----------------- | --------------------------------------------- | ------------------------ | -------------- |
| Translation       | $\begin{bmatrix}I&t\end{bmatrix}_{3x4}$       | 3                        | Orientation    |
| Rigid (Euclidean) | $\begin{bmatrix}R&t\end{bmatrix}_{3x4}$       | 6                        | Length         |
| Similarity        | $\begin{bmatrix}s.R&t\end{bmatrix}_{3x4}$     | 7                        | Angles         |
| Affine            | $\begin{bmatrix}A\end{bmatrix}_{3x4}$         | 12                       | Parallelism    |
| Projective        | $\begin{bmatrix}\tilde{H}\end{bmatrix}_{4x4}$ | 15                       | Straight Lines |

[[1. 3D-Translation]]
[[2. 3D-Rotation]]
[[3. 3D-Scaled Rotation]]
[[4. 3D-Affine]]
[[5. 3D-Projective]]


## (2.1.3) 3D Rotations
### Euler Angles
Generally not recommended to use because
- Result depends on order of rotation (along x or y or z plane)
- 9 parameters$\implies$ not memory efficient
- Can suffer from [[Gimbal Lock]]

See this [[2. 3D-Rotation|Page]] to see how to implement Euler angles for 3D rotation, or this [[Davenport Rotation |Page]] for more detailed explanation.

### Axis-Angle Representation
We can representation a 3D-rotation by $(\vec{e}, \theta)$ where 
- $\vec{e}$ is the 3D vector of the axis of rotation
- $\theta$ is the angle of rotation about that axis

This method is generally a good method for representing 3D rotation, by combining with [[Axis-Angle Representation#Rodrigues Formula |Rodrigues Formula]]
[[Axis-Angle Representation |Read More]]

### Quarternions
This is a 4D-vector, made of 3 imaginary values, and 1 real values: $a + b.\hat{i} + c.\hat{j} + d.\hat{k}$

Although they can be quite hard to understand, they are better suited for representing continuous 3D rotation values such as for [[Quarternions in Rotation#Slerp |Slerp]].

You can read more here:
- [[Quarternions in Rotation]]
- [[Quarternions Math]]

### Conclusion
- Axis-Angle representation is minimal and easy to understand.
- Quarternions are better of keeping track of smoothly moving camera and interpolation

## (2.1.4) 3D to 2D Projection

- [[Orthographic Projection]]
- [[Scaled Orthographic Projection]]
- [[Para-Perspective Projection]]
- [[Perspective Projection]]

The most commonly used projection is perspective.

Scaled orthography can be used as approximate for long focal length lenses, and objects with shallow depth relative to distance.

Para-Perspective can be used as an approximate between scaled orthography & perspective, mainly for computational efficiency purpose.

### Camera Model
A camera can be represented in computer vision using a camera matrix K, made of camera intrinsics & camera extrinsics.

Read more at
- [[Camera Matrix]]
- [[Camera Intrinsics]]
- [[Camera Extrinsics]]
- [[Focal Length]]
- [[Mapping from one camera to another]]

### Depth Representation
When projecting from 3D world coordinates to 2D camera coordinates, we have to take into account of how to project the depth.
There are 2 main ways to achieve this
- [[Inverse Depth]]
- [[Projective Depth]]

## (2.1.5) Lens Distortion
Many wide-angle lenses have noticeable radial distortion which can manifest itself as visible curvature in the projection of straight lines on an image.
Thus, we need to take into account of these distortion and 'fix' them when modelling our camera.

The common distortions are 
- [[Radial Distortion]]
- [[Tangential Distortion]]
- [[Fisheye Distortion]]
- [[Spline Distortion]]
- [[Non-Central Projections]]

