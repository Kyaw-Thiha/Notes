# Sky Model (Computer Vision / Graphics)
 #cv/light/sky-model #graphics

## Core Concept
- **Sky models** describe the **distribution of natural daylight illumination** across the sky dome.  
- Unlike simple **directional lights** (sun) or **ambient terms**, sky models capture the **spatial variation** of skylight due to atmosphere scattering.  
- Widely used in **outdoor computer vision (CV)**, **rendering**, and **environment simulation**.  

---

## Main Properties
- **Physically-based approximations**: Derived from atmospheric scattering theory.  
- **Accounts for**:  
  - Sun position (azimuth, elevation).  
  - Turbidity (haze/air clarity).  
  - Ground reflectance.  
- **Outputs**: Radiance distribution $L(\omega)$ over the hemisphere (sky dome).  
- **Continuous model**: Smooth function rather than discrete map.  

---

## Mathematical Formulation
- General form of sky radiance:  

  $$
  L(\omega) = f(\omega, \theta_s, \phi_s, T)
  $$

  where:  
  - $\omega$: viewing direction.  
  - $(\theta_s, \phi_s)$: solar elevation and azimuth.  
  - $T$: turbidity (atmospheric clarity).  
  - $f$: model-specific function (Preetham, Hosek-Wilkie, etc.).  

- **Irradiance from sky model** at surface point $x$:  

  $$
  E(x) = \int_{\Omega} L(\omega) \cos\theta \, d\omega
  $$

---

## Major Sky Models
1. **Preetham Model (1999)**  
   - Analytical daylight model (based on Perez model).  
   - Simple, widely used in early graphics.  
   - Inputs: sun position, turbidity.  

2. **Hosek-Wilkie Model (2012)**  
   - More accurate, especially near the sun.  
   - Handles atmospheric scattering better.  
   - Often used in modern rendering engines.  

3. **CIE Standard Sky Models**  
   - Defined by International Commission on Illumination (CIE).  
   - Includes multiple standard distributions (clear sky, overcast, etc.).  

---

## ASCII/LaTeX Diagram

```
        Sky Dome with Sun Position

                 ☼  (Sun)
              .-' | `-.
           .-'    |    `-.
        .-'       |       `-.
       /          |          \
      /           |           \
     /            |            \
    -----------------------------
            Ground Plane

L(ω) varies with direction ω depending on sun position and turbidity.
```

---

## Applications in CV
- **Autonomous Driving**: Outdoor lighting simulation for robust vision.  
- **Photometric Stereo Outdoors**: Use sky models for natural illumination constraints.  
- **Scene Relighting**: Consistent outdoor lighting for AR/VR.  
- **Inverse Rendering**: Recovering scene geometry/material under real sky conditions.  

---

## Limitations
- **Approximations**: Cannot fully capture complex cloud patterns or dynamic weather.  
- **Single-Point Validity**: Assumes uniform ground and atmosphere around the scene.  
- **Requires parameters**: Sun position + turbidity must be known or estimated.  

---

## Key Takeaways
- Sky models = **analytic descriptions of skylight distribution**.  
- Provide more realism than ambient/directional lights for outdoor scenes.  
- Common models: **Preetham, Hosek-Wilkie, CIE standard skies**.  
- Essential in **outdoor CV tasks** and **physically-based rendering**.  

---
## See Also
- [[Lighting Models]]