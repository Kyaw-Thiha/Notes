# Tangential Distortion
 #cv/camera/distortion/tangential 

Tangential distortion is another form of lens distortion that arises when the lens is not perfectly aligned with the image sensor. 

Unlike **radial distortion** (which displaces pixels symmetrically towards or away from the image center), tangential distortion introduces **asymmetric shifts** due to decentering or tilting of the lens elements.

---

## 1. Basic Concept

Tangential distortion occurs when:
- The lens is slightly **decentered** (its optical axis does not pass through the exact center of the image sensor).
- The lens is slightly **tilted** with respect to the sensor plane.

As a result, image points are displaced along both $x$ and $y$ directions in a way that depends on their radial position. Unlike radial distortion, which is rotationally symmetric, tangential distortion introduces **directional warping**.

---

## 2. Mathematical Model

Tangential distortion is typically modeled as an additional correction applied to the normalized image coordinates $(x_c, y_c)$:

$$
\hat{x}_c = x_c + \big(2 p_1 x_c y_c + p_2(r_c^2 + 2x_c^2)\big)
$$

$$
\hat{y}_c = y_c + \big(p_1(r_c^2 + 2y_c^2) + 2 p_2 x_c y_c \big)
$$

where:
- $r_c^2 = x_c^2 + y_c^2$ (squared radial distance from the optical axis),
- $p_1, p_2$ are **tangential distortion parameters**,
- $(\hat{x}_c, \hat{y}_c)$ are the corrected (distorted) coordinates.

---

## 3. Origin and Brown–Conrady Model

The **Brown–Conrady distortion model** (Brown, 1966) incorporates both:
- **Radial distortion** (polynomial model with $\kappa_1, \kappa_2, \dots$),
- **Tangential distortion** (using $p_1, p_2$).

This combined model remains the standard in camera calibration pipelines (including OpenCV’s implementation).

---

## 4. Practical Implications

- Tangential distortion tends to **skew rectangular shapes**, making them appear **slanted** or **tilted**.
- Unlike barrel or pincushion distortion, tangential distortion is **not symmetric** across the image.
- It is often smaller in magnitude than radial distortion but must be corrected for accurate **3D reconstruction** and **camera calibration**.

---

## 5. Parameter Estimation

Tangential distortion parameters ($p_1, p_2$) are estimated during camera calibration by minimizing reprojection error:

1. Capture images of a calibration pattern (checkerboard, circles grid, etc.).
2. Extract feature correspondences (pattern corners).
3. Solve for intrinsic parameters, radial distortion ($\kappa_i$), and tangential distortion ($p_1, p_2$) jointly.
4. Use nonlinear optimization (e.g., Levenberg–Marquardt) to minimize error.

---

## 6. Correction

Once parameters are known:
- Apply inverse distortion mapping to shift distorted pixels back to their ideal positions.
- In practice, undistortion is handled using image warping functions (e.g., OpenCV’s `cv::undistort`).

---
## 7. Summary

Tangential distortion is a **decentering effect** caused by misalignment of the lens with the sensor. It is modeled using parameters $p_1, p_2$ that correct asymmetric shifts in pixel coordinates. Although smaller than radial distortion, tangential distortion plays a significant role in precision applications, and modern calibration methods always account for it alongside radial terms.

---
**Next Pages**:  
- [[Radial Distortion]]  
- [[Fisheye Distortion]]  
- [[Spline Distortion]]
- [[Non-Central Projections]]
