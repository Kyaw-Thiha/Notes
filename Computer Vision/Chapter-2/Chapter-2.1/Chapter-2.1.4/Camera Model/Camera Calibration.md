# Camera Calibration  
#cv/camera/calibration   

## Core Concept  
- **Camera calibration** is the process of estimating the parameters of the camera model so that 3D world points can be accurately mapped to 2D image points.  
- Calibration ensures that computer vision algorithms account for the true geometry and optics of the camera.  

---

## Why Calibration Matters  
- Raw images are distorted due to lens effects and imperfect sensor alignment.  
- Calibration recovers **intrinsic** and **extrinsic** parameters to correct for these.  
- Applications:  
  - 3D reconstruction  
  - Augmented reality  
  - Multi-view geometry (stereo, SfM)  
  - Robotics (pose estimation, localization)  

---

## Parameters to Estimate  

### Intrinsic Parameters (camera internals)  
- [[Focal Length]] $f_x, f_y$  
- [[Principal Point]] $(c_x, c_y)$  
- [[Skew]] $\gamma$ (rare in modern sensors)  
- [[Distortion Models]]  
  - Radial distortion (barrel/pincushion)  
  - Tangential distortion (lens misalignment)  
  - Chromatic aberration (per-color distortion)  

### Extrinsic Parameters (camera pose)  
- [[Rotation Matrix]] $R$  
- [[Translation Vector]] $t$  
- Defines transformation from **world coordinates → camera coordinates**.  

---

## Mathematical Formulation  

1. **Projection equation**  

   $$
   s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} 
   = K \, [R \,|\, t] \, \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}
   $$  

   where:  
   - $(u,v)$ = pixel coordinates  
   - $(X,Y,Z)$ = 3D world point  
   - $K$ = intrinsic matrix  
   - $R, t$ = extrinsic parameters  
   - $s$ = projective depth  

2. **Distortion correction**  
   - Apply inverse of distortion model before projection.  

---

## Calibration Techniques  

- **Checkerboard (planar patterns)**  
  - Capture multiple images of known calibration target.  
  - Solve for intrinsics and extrinsics (Zhang’s method, 2000).  

- **3D calibration rigs**  
  - Use known 3D structures for higher precision.  

- **Self-calibration**  
  - Infer parameters directly from image sequences (no target).  

- **Online calibration**  
  - Update parameters dynamically in SLAM / robotics.  

---

## ASCII/LaTeX Diagram  

### Checkerboard Calibration  
```
   Camera ---->   [Checkerboard with known square size]
                     +---+---+---+
                     |   |   |   |
                     +---+---+---+
                     |   |   |   |
                     +---+---+---+
   Extract corners → match 2D ↔ 3D → solve K, R, t
```

---

## Applications in CV  
- **Stereo vision**: requires calibrated pair of cameras.  
- **AR/VR**: precise intrinsics needed for overlaying graphics.  
- **Robotics**: essential for mapping and localization.  
- **Photogrammetry**: reconstruct accurate 3D models.  

---

## Key Takeaways  
- Calibration links **3D world geometry** to **2D image measurements**.  
- Requires estimating intrinsics (K) and extrinsics (R, t).  
- Correcting lens distortions is part of the process.  
- Commonly performed using **checkerboard patterns** and algorithms like Zhang’s method.  

---

## See Also  
- [[Camera Matrix]]  
- [[Camera Intrinsics]]  
- [[Camera Extrinsics]]  
- [[Distortion Models]]  
- [[Focal Length]]  

