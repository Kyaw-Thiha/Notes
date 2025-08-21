# Point Light Source 
 #cv/light/point #graphics

## Core Concept
- A **point light source** is an idealized light emitter that radiates energy from a **single point in space**.  
- It has **no size** and **no shape**, only a position and an emission pattern.  
- Used extensively in computer vision (CV) and graphics to simplify lighting models.  

---

## Main Properties
- **Position-only**: Defined by its location in 3D space, $(x, y, z)$.  
- **No area**: Unlike area lights, point lights have no surface.  
- **Emission pattern**:  
  - *Isotropic*: emits equally in all directions.  
  - *Anisotropic*: emission varies with direction (modeled via intensity function $I(\omega)$).  

---

## Mathematical Formulation
- **Radiant Intensity** (directional emission):  

  $$
  I(\omega) = \frac{d\Phi}{d\omega}
  $$

  where $\Phi$ is total flux (power), $\omega$ direction.  

- **Irradiance at a point $p$** (receiving surface):  

  $$
  E(p) = \frac{I(\omega)}{r^2} \cos\theta
  $$

  - $r$: distance between light source and point $p$  
  - $\theta$: angle between incoming direction and surface normal  
  - $\frac{1}{r^2}$: inverse-square falloff  

- **Radiance received** is derived from intensity and distance, propagating along rays.  

---

## ASCII/LaTeX Diagram

```
               * Point Light Source
              /|\ 
             / | \
            /  |  \
           /   |   \
          ↓    ↓    ↓
   Surface Point (p) with normal (n)

Irradiance at p:  E(p) = I(ω)/r² · cosθ
```

LaTeX version:

$$
E(p) = \frac{I(\omega)}{r^2} \cos\theta
$$

---

## Applications in CV
- **Photometric Stereo**: Point light models simplify normal recovery under controlled lighting.  
- **Scene Relighting**: Easy to manipulate point lights for synthetic augmentation.  
- **Calibration**: Many CV setups approximate lamps/LEDs as point sources.  
- **Rendering & Simulation**: Core component in physically-based rendering pipelines.  

---

## Limitations
- **Unrealistic**: Real lights always have finite area → point light ignores shadows/softness.  
- **Infinite Intensity at Source**: Since no area, physical intensity becomes undefined at $r=0$.  
- **No Penumbra**: All shadows are perfectly sharp (no soft edges).  

---

## Key Takeaways
- Point light source = **simplest light model** (all power concentrated at one position).  
- Irradiance on surfaces follows **inverse-square law**.  
- Very useful for **CV theory, calibration, and simplified rendering**, but limited realism.  

---
## See Also
- [[Lighting Models]]
- [[Radiant Intensity]]
- [[Radiance]]
- [[Irradiance]]