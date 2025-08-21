# Environment Map (Computer Vision / Graphics)
 #cv/light/environment-map #graphics

## Core Concept
- An **environment map** represents **lighting from all directions** surrounding a scene.  
- Instead of modeling discrete light sources, it encodes illumination as a function on the **sphere of directions** around a point.  
- Widely used in computer vision (CV) and graphics for **image-based lighting (IBL)**, relighting, and material appearance modeling.  

---

## Main Properties
- **Representation**: A function $L(\omega)$ over directions $\omega$ on the unit sphere.  
- **Encodes radiance**: Each direction stores incoming radiance from the environment.  
- **Infinite distance assumption**: Light sources are assumed to be infinitely far, so direction matters but position does not.  

---

## Common Parameterizations
- **Spherical Map**: Lat-long projection of the sphere.  
- **Cube Map**: 6 images mapped to cube faces, then projected to directions.  
- **Hemisphere Map**: Used for ground-plane or upper-hemisphere lighting.  

---

## Mathematical Formulation
- **Irradiance from environment map at surface point $x$:**

  $$
  E(x) = \int_{\Omega} L(\omega) \cos\theta \, d\omega
  $$

  where:  
  - $L(\omega)$: incoming radiance from direction $\omega$  
  - $\theta$: angle between $\omega$ and surface normal  
  - $\Omega$: hemisphere above the surface  

- **Reflection using BRDF ($f_r$):**

  $$
  L_o(x, \omega_o) = \int_{\Omega} f_r(x, \omega_i, \omega_o) \, L(\omega_i) \cos\theta \, d\omega_i
  $$

---

## ASCII/LaTeX Diagram

```
       Environment Map as Surrounding Sphere

            +-------------------+
           /                     \
          /                       \
         |                         |
         |       (Scene)           |
         |           o <- Object   |
          \                       /
           \                     /
            +-------------------+

 Every direction ω corresponds to a pixel in the environment map
 storing incoming radiance L(ω).
```

---

## Applications in CV
- **Image-Based Lighting (IBL)**: Real-world captures (HDR panoramas) provide realistic lighting for virtual objects.  
- **Relighting**: Enables consistent illumination when compositing real + synthetic elements.  
- **Material Estimation**: Used in inverse rendering to recover BRDF under natural lighting.  
- **Scene Understanding**: CV systems use environment maps to reason about outdoor/indoor illumination.  

---

## Limitations
- **Assumes distant lighting**: Cannot model nearby lights well.  
- **Resolution trade-off**: High-resolution maps needed for sharp shadows.  
- **Single point of view**: Usually defined for one scene location, may not hold across large spaces.  

---

## Key Takeaways
- Environment map = **global lighting description** (radiance as a function of direction).  
- Bridges **captured real-world illumination** and **physically based rendering**.  
- Essential for realistic **relighting, material capture, and IBL in CV/graphics**.  

---
## See Also
- [[Lighting Models]]