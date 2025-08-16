# Mapping from One Camera to Another
 #cv/camera/mapping

![Camera Mapping](camera-mapping.png)
When two cameras capture images of the same 3D scene from different **positions** or **orientations**, we want to understand how a point in one image relates to its corresponding point in the other image. 
This mapping depends on the **camera matrices**, the scene geometry, and sometimes additional assumptions.

---

## 1. General Projection with Camera Matrices

Each camera has a **$4 \times 4$ projection matrix**:

$$
\tilde{x}_0 \sim \tilde{K}_0 E_0 p = \tilde{P}_0 p
$$

- $p = (X, Y, Z, 1)$: 3D world point in homogeneous coordinates.  
- $\tilde{K}_0$: intrinsic calibration matrix.  
- $E_0$: extrinsic matrix (camera pose).  
- $\tilde{P}_0 = \tilde{K}_0 E_0$: full camera matrix.  
- $\tilde{x}_0$: homogeneous pixel coordinates in camera 0.

If we know the **depth** (disparity) $d_0$ for a pixel $\tilde{x}_0$, we can recover $p$:

$$
p \sim E_0^{-1} \tilde{K}_0^{-1} \tilde{x}_0
$$

Then, we can project it into another camera:

$$
\tilde{x}_1 \sim \tilde{K}_1 E_1 p
= \tilde{K}_1 E_1 E_0^{-1} \tilde{K}_0^{-1} \tilde{x}_0
= \tilde{P}_1 \tilde{P}_0^{-1} \tilde{x}_0
= M_{10} \tilde{x}_0
$$

where $M_{10}$ is the **inter-camera mapping** matrix.

---

## 2. Problem: Missing Depth

In typical photographs, the **depth $d_0$ is unknown** for each pixel. Without it, we cannot directly compute the 3D position $p$. However, there are **two important special cases** where depth is not required:

---

## 3. Case 1: Planar Scene (Homography from a Plane)

If all points lie on a single plane (think of image of a wall) with equation:

$$
\hat{n}_0 \cdot p + c_0 = 0,
$$

we can substitute this plane equation into the projection model. In this case, the mapping reduces to:

$$
\tilde{x}_1 \sim \tilde{H}_{10} \tilde{x}_0
$$

- $\tilde{H}_{10}$: a $3 \times 3$ **homography matrix**.  
- $\tilde{x}_0, \tilde{x}_1$: 2D homogeneous pixel coordinates.

This shows why an **8-parameter homography** is a valid model for planar scene alignment, such as:

- **Image mosaics**  
- **Stitching planar textures**  

---

## 4. Case 2: Pure Camera Rotation

If the cameras differ only by a **rotation** ($t_0 = t_1$), then:

$$
\tilde{x}_1 \sim K_1 R_1 R_0^{-1} K_0^{-1} \tilde{x}_0
= K_1 R_{10} K_0^{-1} \tilde{x}_0
$$

where $R_{10}$ is the relative rotation.

- This too reduces to a **$3 \times 3$ homography**.  
- Assumptions:
  - Calibration matrices $K_0, K_1$ are known.  
  - Aspect ratios and principal points are set (often using the simplified form of $K$).

This case is the foundation of **image stitching** for panoramas, where the camera rotates about its optical center.

---

## 5. Summary

- **General mapping** requires depth (Eq. 2.70).  
- **Planar scene** → reduces to a $3 \times 3$ homography (Eq. 2.71).  
- **Pure rotation** → also reduces to a $3 \times 3$ homography (Eq. 2.72).  
- Both cases avoid needing per-pixel depth and are widely used in **multi-view geometry** and **image alignment**.  

✅ Thus, mapping between cameras is either a **general 3D transformation** (depth required) or simplifies to a **homography** under special conditions (planarity or pure rotation).

## See Also
- [[Camera Matrix]]
- [[2. 3D-Rotation]]