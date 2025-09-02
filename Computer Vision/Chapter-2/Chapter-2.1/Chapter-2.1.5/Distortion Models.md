# Distortion Models  
#cv/camera/distortion   

## Core Concept  
- **Lens distortion** occurs when the actual mapping from 3D rays to 2D image points deviates from the ideal pinhole or thin-lens camera model.  
- Distortions are most noticeable in wide-angle or cheap lenses, where straight lines appear curved or shifted.  
- Accurate modeling and correction are essential for **camera calibration**, **3D reconstruction**, and **image rectification**.  

---

## Common Distortion Types  

### [[Radial Distortion]]  
- Caused by imperfections in lens curvature.  
- Straight lines bow outward (**barrel**) or inward (**pincushion**).  
- Modeled using polynomial terms in radius $r$:  

  $$
  x_{\text{dist}} = x (1 + k_1 r^2 + k_2 r^4 + k_3 r^6 + \dots)
  $$  

### [[Tangential Distortion]]  
- Caused by lens not being perfectly parallel to the sensor.  
- Shifts image points asymmetrically.  
- Modeled with coefficients $p_1, p_2$:  

  $$
  x_{\text{dist}} = x + [2p_1xy + p_2(r^2 + 2x^2)]
  $$  
  $$
  y_{\text{dist}} = y + [p_1(r^2 + 2y^2) + 2p_2xy]
  $$  

### [[Fisheye Distortion]]  
- Extreme wide-angle lenses (field of view > 180°).  
- Rays mapped with nonlinear projection (e.g., equidistant, stereographic, equisolid angle).  
- Requires specialized fisheye calibration models.  

### [[Spline Distortion]]  
- Flexible distortion correction using splines or piecewise functions.  
- More accurate for complex lens systems.  
- Often used in high-precision photogrammetry.  

### [[Non-Central Projections]]  
- Not all systems can be modeled as rays passing through a **single center of projection**.  
- Examples: catadioptric (lens + mirror), multi-camera arrays.  
- Require explicit **ray–pixel mapping functions** instead of simple formulas.  

---

## ASCII/LaTeX Diagram  

### Barrel vs Pincushion Distortion  
```
   Ideal Grid        Barrel Distortion      Pincushion Distortion
   +---+---+---+     +---+---+---+         +---+---+---+
   |   |   |   |     |  )|   |(  |         |(  |   |  )|
   +---+---+---+     +---+---+---+         +---+---+---+
   |   |   |   |     |   |   |   |         |   |   |   |
   +---+---+---+     +---+---+---+         +---+---+---+
```

---

## Applications in CV  
- **Camera calibration**: distortion parameters are estimated alongside intrinsics.  
- **Image rectification**: undistort images for measurements.  
- **Robotics / SLAM**: accurate geometry requires distortion correction.  
- **Rendering & simulation**: distortion models are used to emulate realistic lenses.  

---

## Key Takeaways  
- Distortion = deviation from ideal pinhole projection.  
- Main types: **radial, tangential, fisheye, spline, non-central**.  
- Correction requires estimating distortion coefficients during calibration.  
- Some systems cannot be corrected with a simple pinhole assumption → need ray-mapping.  

---

## See Also  
- [[Camera Calibration]]  
- [[Camera Matrix]]  
- [[Optics]]  
- [[Chromatic Aberration]]  

