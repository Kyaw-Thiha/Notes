# HDR Environment Maps (Computer Vision / Graphics)
 #cv/light/hdr #graphics

## Core Concept
- An **HDR environment map** is an environment map stored in **high dynamic range (HDR)** format, capturing the full range of scene illumination.  
- Unlike standard LDR (low dynamic range) images, HDR maps preserve both very bright (e.g., sun, lamps) and very dark details.  
- Essential in **image-based lighting (IBL)**, where accurate light intensity ratios matter for realistic rendering and CV analysis.  

---

## Main Properties
- **Radiance-accurate**: Stores scene radiance values, not just display-ready pixel intensities.  
- **File formats**: Commonly `.hdr`, `.exr` (floating-point values).  
- **Dynamic range**: Captures orders of magnitude of brightness (e.g., sunlight vs shadow).  
- **Tone mapping**: HDR is usually tone-mapped for visualization but raw values used in computation.  

---

## Mathematical Formulation
- **Irradiance from HDR environment map:**

  $$
  E(x) = \int_{\Omega} L_{HDR}(\omega) \cos\theta \, d\omega
  $$

  - $L_{HDR}(\omega)$: radiance values stored in HDR map.  
  - $\theta$: angle between direction $\omega$ and surface normal.  
  - Ensures that very bright sources (like the sun) contribute proportionally.  

- **Rendering equation under HDR map:**

  $$
  L_o(x, \omega_o) = \int_{\Omega} f_r(x, \omega_i, \omega_o) \, L_{HDR}(\omega_i) \cos\theta \, d\omega_i
  $$

---

## ASCII/LaTeX Diagram

```
       HDR Environment Map (Lat-Long Projection)

       +--------------------------------+
       |    bright sun (high radiance)  |
       |                                |
       |   sky gradient                 |
       |                                |
       |   dark ground / buildings      |
       +--------------------------------+

 Brightness values preserved numerically,
 not clamped → accurate illumination.
```

---

## Applications in CV
- **Relighting & Augmented Reality**: Insert synthetic objects into real-world scenes with correct shadows and reflections.  
- **Material Capture**: Accurately recover reflectance (BRDF/BSDF) under real-world lighting.  
- **Scene Understanding**: HDR sky models improve outdoor illumination reasoning.  
- **Inverse Rendering**: Use HDR maps as constraints for recovering geometry + reflectance.  

---

## Limitations
- **Acquisition cost**: Requires bracketed exposures or specialized HDR cameras.  
- **Large storage**: Floating-point maps are heavier than LDR images.  
- **Processing overhead**: Needs tone mapping for display, but raw HDR values for computation.  

---

## Key Takeaways
- HDR environment maps = **physically accurate lighting representation**.  
- Crucial for **realism in IBL, relighting, and CV material studies**.  
- Unlike LDR maps, they maintain correct **intensity ratios** between light sources.  
