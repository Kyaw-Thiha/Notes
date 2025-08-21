# Camera Matrix
 #cv/camera/camera-matrix 

The **camera matrix** combines both the **intrinsic parameters** (camera internals) and **extrinsic parameters** (camera pose in the world) into a single projection model. It allows mapping **3D world points** directly into **2D pixel coordinates**.

---
## 1. Definition

The **$3 \times 4$ camera matrix** is defined as

$$
P = K [R \;|\; t]
$$

where:

- $K$: **intrinsic calibration matrix** (focal lengths, skew, principal point).  
- $R$: **rotation matrix** (camera orientation in the world).  
- $t$: **translation vector** (camera position in the world).  

This matrix projects a 3D point $p_w$ in world coordinates to a pixel coordinate $\tilde{x}_s$:

$$
\tilde{x}_s = P \, p_w
$$

with $\tilde{x}_s$ expressed in **homogeneous image coordinates**.

---
## 2. Homogeneous Form

- A 3D point in the world:  
  $$
  p_w = (x_w, y_w, z_w, 1)^T
  $$
- The camera matrix $P$ transforms this into homogeneous image coordinates:  
  $$
  \tilde{x}_s = (x_s, y_s, 1)^T \sim P p_w
  $$
  where $\sim$ denotes equality up to scale.

After projection, coordinates are normalized by dividing by the last element to recover standard pixel coordinates.

---
## 3. Alternative $4 \times 4$ Camera Matrix

Sometimes, it is useful to work with a **full-rank 4×4 matrix** to preserve invertibility:

$$
\tilde{P} =
\begin{bmatrix}
K & 0 \\
0^T & 1
\end{bmatrix}
\begin{bmatrix}
R & t \\
0^T & 1
\end{bmatrix}
= \tilde{K} E
$$

- $\tilde{K}$: extended calibration matrix.  
- $E$: Euclidean rigid-body transformation (rotation + translation).

This allows direct mapping from homogeneous 3D coordinates to **screen coordinates with disparity**:

$$
x_s = (x_s, y_s, 1, d)^T \sim \tilde{P} \, \bar{p}_w
$$

- $\bar{p}_w$: homogeneous 3D world point.  
- $d$: disparity or depth value.  

---

## 4. Depth Representations

- **Inverse depth (disparity):**  
  $d = \frac{1}{z}$, decreases as object moves further away.  

- **Projective depth (relative to a reference plane):**  
  Encodes parallax relative to a chosen plane, can be positive or negative depending on whether point is in front or behind the plane.

These interpretations are important for stereo vision and structure-from-motion.

Read more at
- [[Inverse Depth]]
- [[Projective Depth]]

---

## 5. Key Points

- The **camera matrix $P$** is the central object in projective geometry for vision.  
- It has **11 degrees of freedom** (intrinsics: ~5, extrinsics: 6).  
- Projection:  
  $$
  p_w \xrightarrow{[R|t]} p_c \xrightarrow{K} \tilde{x}_s
  $$
- Simplifications (zero skew, centered principal point) reduce the number of unknowns.  
- The extended $4 \times 4$ form $\tilde{P}$ is convenient in computer graphics and stereo geometry, as it retains disparity/depth explicitly.

---

## 6. Summary

- **$P = K [R|t]$** projects **world → image**.  
- Combines **intrinsics** (sensor model) and **extrinsics** (camera pose).  
- A 3×4 projective matrix, often extended to 4×4 for invertibility.  
- Encodes all transformations needed to go from **3D scene geometry** to **2D image pixels**.

## See Also
- [[Camera Intrinsics]]
- [[Camera Extrinsics]]