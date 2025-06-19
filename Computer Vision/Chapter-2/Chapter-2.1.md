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
- Smaller change in rotation can lead to large change of Euler angle --> Cannot move smoothly along rotation
- Thus, can lead to 

See this [[2. 3D-Rotation|Page]] to see how to implement Euler angles for 3D rotation