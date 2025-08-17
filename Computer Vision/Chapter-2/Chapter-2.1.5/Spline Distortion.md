# Spline Distortion
 #cv/camera/distortion/spline

Spline distortion models are used when traditional polynomial-based models (radial, tangential, or fisheye) fail to accurately capture the complex distortions introduced by certain lenses. 

Instead of relying on a fixed analytical formula, spline distortion employs **piecewise polynomial functions (splines)** to flexibly describe irregular distortions across the image.

---

## 1. Motivation

- **Polynomial models (radial + tangential)**:  
  Work well for most standard and wide-angle lenses, but become inaccurate for:
  - Extremely wide-angle or fisheye lenses,
  - Lenses with manufacturing imperfections,
  - Non-standard or anamorphic optics.
  
- **Spline-based models**:  
  Provide a **non-parametric, flexible approach** that can adapt to complex, spatially varying distortions. This makes them suitable when distortions do not follow simple radial or tangential patterns.

---

## 2. Basic Concept

A spline distortion model defines a smooth mapping between **ideal (undistorted) coordinates** $(x_u, y_u)$ and **observed (distorted) coordinates** $(x_d, y_d)$ using spline functions.

General idea:

$$
x_d = S_x(x_u, y_u) \\
y_d = S_y(x_u, y_u)
$$

where:
- $S_x, S_y$ are 2D spline functions defined over a grid of control points,
- Control points are optimized during calibration to minimize reprojection error.

---

## 3. Types of Splines Used

- **B-splines (basis splines)**:  
  Flexible, smooth, and widely used for geometric modeling.
  
- **Thin-plate splines (TPS)**:  
  Often used in image warping, providing smooth deformations across the entire image plane.
  
- **Cubic splines**:  
  Simpler and computationally efficient, though less flexible than TPS.

---

## 4. Calibration and Estimation

1. **Calibration pattern** (checkerboard or dot grid) is imaged with the distorted lens.
2. Correspondences between known 3D points and observed 2D projections are extracted.
3. A **spline-based mapping** is fitted to minimize the reprojection error:
   $$
   E = \sum_i \| (x_d^i, y_d^i) - (S_x(x_u^i, y_u^i), S_y(x_u^i, y_u^i)) \|^2
   $$
4. Optimization adjusts control points of the spline grid.

---

## 5. Advantages

- Can model **irregular distortions** that are not well represented by polynomials.
- Particularly useful when:
  - The lens does not have a **single center of projection**,
  - Manufacturing defects or design choices cause **localized distortions**,
  - Large field-of-view distortions require more flexibility.

---

## 6. Applications

- **Wide-angle and fisheye calibration** when polynomial models fail.
- **Medical imaging optics**, where lenses may introduce complex warping.
- **High-precision photogrammetry**, where even small distortions must be accounted for.
- **Computer vision research**, especially for experimental or custom optics.

---

## 7. Limitations

- Requires more calibration data than simple polynomial models.  
- Computationally more expensive.  
- Risk of **overfitting** if too many spline control points are used.

---

## 8. Summary

Spline distortion models provide a flexible, data-driven way to model complex lens distortions. By replacing rigid polynomial equations with spline mappings, they can handle irregular, non-central, or highly non-linear distortions. While more computationally demanding, they are indispensable for applications requiring **sub-pixel geometric accuracy** with unusual lens systems.

---
**Next Pages**:  
- [[Radial Distortion]]  
- [[Tangential Distortion]]  
- [[Fisheye Distortion]]  
- [[Non-Central Projections]]
