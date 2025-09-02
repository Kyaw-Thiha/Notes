# Optics  
#cv/optics  

![[optics.png]]
## Core Concept  
- **Optics** studies how light interacts with lenses before reaching the sensor.  
- In computer vision, we often simplify lenses as pinholes.  
- To handle **focus, exposure, depth of field, and aberrations**, we need lens-based models.  

---

## Thin Lens Model  
- Simplest approximation: **thin lens** with focal length $f$.  
- Relationship between object distance $z_o$, image distance $z_i$, and focal length:  

  $$
  \frac{1}{z_o} + \frac{1}{z_i} = \frac{1}{f}
  $$

- If $z_o \to \infty$, then $z_i = f$.  
  - This explains why a lens of focal length $f$ acts like a **pinhole** at distance $f$ from the sensor.  

---

## Depth of Field & Circle of Confusion  
- **Focal plane**: the image plane where objects at $z_o$ appear sharp.  
- If shifted by $\Delta z_i$, the image becomes blurred.  
- **Circle of Confusion (CoC)** $c$: blur size when misfocused.  
  - Derived via similar triangles.  
  - Depends on $\Delta z_i$, original focus distance $z_i$, and aperture diameter $d$.  
- **Depth of Field (DoF)**: scene range where blur $c$ is within acceptable limits.  

---

## Aperture and f-number  
- Aperture diameter $d$ controls how much light enters.  
- **f-number (f-stop)**:  

  $$
  N = \frac{f}{d}
  $$

- Common values: f/1.4, f/2, f/2.8, … f/22.  
- Each full stop changes exposure by factor of 2 (multiplying/dividing area by 2).  
- **Smaller $N$ (wider aperture)** → shallow DoF, brighter image.  
- **Larger $N$ (narrow aperture)** → deeper DoF, dimmer image.  

---

## Geometric Aberrations (Seidel’s Five)  
Even thin lenses are imperfect:  
- **Spherical aberration** – edges focus differently than center.  
- **Coma** – off-axis points become comet-shaped.  
- **Astigmatism** – different focus in vertical/horizontal planes.  
- **Curvature of field** – flat objects appear curved in focus.  
- **Distortion** – straight lines bend (barrel/pincushion).  

---

## ASCII/LaTeX Diagram  

### Thin Lens Imaging
```
Object Plane (z_o)         Lens (f)                Image Plane (z_i)
     |                       |                           |
     *   ----->     | )   )  |  ----->       *
     |                       |                           |

Formula: 1/z_o + 1/z_i = 1/f
```

### Circle of Confusion
```
  Object Point (misfocused)
       \
        \         Aperture (d)
         \      /------O------\
          \    |               |
           \---|      Lens     |---   Focal plane shifted Δz_i
                \             /
                 \-----------/
                  Blur spot (circle of confusion, c)
```

---

## Applications in CV  
- **Camera calibration**: lens parameters are needed for accurate 3D reconstruction.  
- **Depth estimation**: defocus blur can provide depth cues.  
- **Image correction**: remove distortions and aberrations.  
- **Rendering**: simulate realistic DoF and lens effects.  

---

## Key Takeaways  
- Thin lens law connects object, image, and focal length.  
- Depth of field depends on **aperture size** and **focus distance**.  
- f-number controls exposure and blur.  
- Real lenses deviate due to **geometric aberrations**, requiring correction.  

---

## See Also  
- [[Chromatic Aberration]]  
- [[Vignetting]]  
- [[Radiance]]  
- [[Irradiance]]  
- [[Focal Length]]

