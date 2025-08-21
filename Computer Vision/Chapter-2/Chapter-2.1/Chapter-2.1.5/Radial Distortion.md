# Radial Distortion
 #cv/camera/distortion/radial

![Fisheye Distortion](radial-fisheye-distortion.png)

Radial distortion is one of the most common types of lens distortions observed in wide-angle lenses. 

It arises when straight lines in the real world do not project as straight lines in the image, breaking the assumption of a **linear projection model** (where homogeneous coordinates undergo linear matrix transformations). 

Correcting radial distortion is crucial for accurate photorealistic reconstructions, mosaics, and any vision system relying on precise geometry.

---
## 1. Basic Concept

Radial distortion occurs when image points are displaced radially (i.e., towards or away from the image center). Depending on the displacement:

- **Barrel distortion**: Points move **towards** the image center, making straight lines bulge outwards.
- **Pincushion distortion**: Points move **away** from the image center, making straight lines curve inwards.

Formally, let $(x_c, y_c)$ be the normalized pixel coordinates after perspective division (before focal scaling and principal point shifting).
In other words, this is done before applying camera intrinsics matrix.

Then the distorted coordinates $(\hat{x}_c, \hat{y}_c)$ are given by:

$$
\hat{x}_c = x_c \big(1 + \kappa_1 r_c^2 + \kappa_2 r_c^4 \big)
$$

$$
\hat{y}_c = y_c \big(1 + \kappa_1 r_c^2 + \kappa_2 r_c^4 \big)
$$

where

$$
r_c^2 = x_c^2 + y_c^2
$$

and $\kappa_1, \kappa_2$ are **radial distortion parameters**.

---
## 2. Brown–Conrady Model

This model (Brown, 1966) is widely used in photogrammetry and computer vision calibration. It includes both:

- **Radial distortion** (dominant component, as shown above).
- **Tangential distortion** (caused by lens decentering).  
  In practice, tangential terms are often ignored due to their instability during estimation (Zhang, 2000).

---
## 3. From Normalized to Pixel Coordinates

After applying distortion, final pixel coordinates are computed by scaling with focal length $f$ and adding the principal point offset $(c_x, c_y)$:

$$
x_s = f \hat{x}_c + c_x
$$

$$
y_s = f \hat{y}_c + c_y
$$

This step converts distorted normalized coordinates into actual pixel positions on the image sensor.

---
## 4. Estimation of Parameters

To use this model, $\kappa_1$ and $\kappa_2$ must be **calibrated**. Common techniques include:

- **Checkerboard calibration** (e.g., Zhang’s method).
- **Feature-based approaches** (aligning known geometric structures).
- **Optimization methods** (minimizing re-projection error using nonlinear least-squares).

---
## 5. Practical Implications

- If distortion is not corrected:
  - **Image mosaics** will exhibit blur due to misalignment during blending.
  - **3D reconstructions** will be inaccurate.
- Correcting radial distortion restores the camera model back to a **linear imager**, allowing intrinsic matrix decomposition and rotation separation to remain valid.

---
## 6. Types of Radial Distortion

- **Barrel distortion**:  
  Straight lines bulge outward. Common in wide-angle lenses.

- **Pincushion distortion**:  
  Straight lines curve inward. More common in telephoto lenses.

---
## 7. Summary

Radial distortion correction is a critical step in camera calibration and computer vision pipelines. The **polynomial model with $\kappa_1, \kappa_2$** captures most practical distortions. While simplified, it provides stable and accurate results in most applications, especially when tangential components are negligible.

---
**Next Pages**:  
- [[Fisheye Distortion]]  
- [[Spline Distortion]]
- [[Tangential Distortion]]
- [[Non-Central Projections]]
- [[Camera Intrinsics]]
- [[Camera Matrix]]
