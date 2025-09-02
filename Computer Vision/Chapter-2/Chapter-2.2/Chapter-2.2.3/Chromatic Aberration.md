# Chromatic Aberration  
#cv/optics/aberration   

![[chromatic_aberration.png]]
## Core Concept  
- **Chromatic aberration** occurs because the **index of refraction varies with wavelength**.  
- Different colors (wavelengths) of light focus at different distances → image blur or color fringes.  
- Two main types:  
  - **Transverse chromatic aberration** → lateral displacement (color edges).  
  - **Longitudinal chromatic aberration** → different focal depths (blur by color).  

---

## Textbook Insight (Szeliski, 2nd ed., Sec. 2.2.3)  
- Simple lenses: red and blue rays have different focal lengths ($f'$).  
- Causes:  
  - Geometric (in-plane) displacement.  
  - Loss of sharp focus (blur).  
- **Transverse aberration** can be modeled as per-color radial distortion → calibrated like distortion correction (Sec. 11.1.4).  
- **Longitudinal aberration** harder to fix → high frequencies attenuated, information lost.  

---

## Compound Lenses & Nodal Points  
- Most modern lenses are **compound lenses** (multiple glass elements).  
- Designed to reduce chromatic and geometric aberrations.  
- Such lenses:  
  - Have a **front nodal point** (entry).  
  - Have a **rear nodal point** (exit).  
  - Useful in **panorama stitching** (rotate around nodal point to avoid parallax).  
- Special cases:  
  - Fisheye and catadioptric (lens + mirrors) **do not have a single nodal point**.  
  - Require explicit mapping functions (pixel ↔ ray).  

---

## ASCII/LaTeX Diagram  

### Chromatic Aberration Effect  
```
                Lens
     Red Ray  /     \   Blue Ray
             /       \
Object * ---/---------\--->
             \       /
              \     /
             Sensor Plane

Red focus: z_i (focal length f)
Blue focus: z_i' (focal length f')
```

- Different colors → different image plane locations ($z_i \neq z_i'$).  

---

## Mathematical Note  
- **Transverse Chromatic Aberration**:  
  Modeled as per-color **radial distortion** function $r(\lambda)$.  

- **Longitudinal Chromatic Aberration**:  
  Different focal lengths $f(\lambda)$ for each wavelength.  
  Blur increases for colors far from the chosen focus.  

---

## Applications in CV  
- **Calibration**: chromatic aberration must be modeled for accurate reconstruction.  
- **Panorama stitching**: nodal point calibration critical for parallax-free capture.  
- **Restoration**: some post-processing can reduce color fringes, but blur is harder to undo.  
- **Synthetic rendering**: adding chromatic aberration improves realism.  

---

## Key Takeaways  
- Chromatic aberration = **wavelength-dependent focusing error**.  
- Two forms: **transverse (color displacement)** and **longitudinal (color blur)**.  
- Compound lenses reduce but don’t eliminate it.  
- Sometimes corrected in software (calibration), but information loss makes full recovery impossible.  

---

## See Also  
- [[Optics]]  
- [[Vignetting]]  
- [[Camera Calibration]]  
- [[Distortion Models]]  


