# Directional Light (Computer Vision / Graphics)
 #cv/light/directional #graphics

## Core Concept
- A **directional light** models illumination from a **very distant source**, such that **all incoming rays are parallel**.  
- Typical example: **sunlight** → distance is so large that angular variation is negligible.  
- Defined by a **direction vector** and **intensity**, but has **no position**.  

---

## Main Properties
- **Infinite distance assumption**: no attenuation with distance.  
- **Uniform direction**: every point in the scene receives light from the same $\omega$.  
- **Units**: described using **irradiance per unit area** or **radiance per direction**.  
- **No inverse-square falloff**: unlike point lights, intensity remains constant across the scene.  

---

## Mathematical Formulation
- **Irradiance from directional light at surface point $p$:**

  $$
  E(p) = L \cos\theta
  $$

  where:  
  - $L$: radiance of the directional light (constant for all points)  
  - $\theta$: angle between light direction and surface normal  

- **Shading relation** (Lambertian):  

  $$
  I(p) \propto \max(0, \cos\theta)
  $$

  → classical diffuse shading under directional light.  

---

## ASCII/LaTeX Diagram

```
    Parallel rays from distant source (sunlight)

   ↓    ↓    ↓    ↓    ↓    ↓
   ↓    ↓    ↓    ↓    ↓    ↓
  ---------------------------
     Surface with normal (n)

Irradiance = L · cosθ (constant across scene)
```

---

## Applications in CV
- **Outdoor Illumination**: Sun approximated as a directional light in many CV tasks.  
- **Photometric Stereo**: Directional lights simplify normal recovery equations.  
- **Inverse Rendering**: Directional light models reduce complexity in relighting.  
- **Synthetic Data Generation**: Directional lights are standard in rendering engines for outdoor scenes.  

---

## Limitations
- **Unrealistic indoors**: Most indoor lights have finite distance/size → point or area light more accurate.  
- **No shadows softness**: Shadows are perfectly sharp unless combined with sky/area light.  
- **Ignores falloff**: Cannot represent near-field lighting accurately.  

---

## Key Takeaways
- Directional light = **parallel rays from infinity** (ideal for sunlight).  
- Provides **uniform, distance-independent illumination**.  
- Fundamental in **outdoor CV**, **diffuse shading models**, and **simplified rendering setups**.  

---
## See Also
- [[Lighting Models]]