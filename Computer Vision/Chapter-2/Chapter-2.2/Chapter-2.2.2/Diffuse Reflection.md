# Diffuse Reflection

![BRDF](./attachments/diffuse_specular.png)

## Description
Diffuse reflection (also called **Lambertian** or **matte reflection**) describes how light scatters uniformly in all directions after striking a rough surface. Unlike specular reflection, it does not produce highlights, but instead results in smooth shading across the surface. It is the dominant phenomenon in most everyday objects (e.g., statues, unpolished wood, painted walls).

Diffuse reflection often imparts the **body color** of the material, since light penetrates slightly, undergoes selective absorption, and then re-emerges.

---

## Key Properties
- **Uniform scattering**: Reflected light intensity is independent of the viewing direction.  
- **Angle dependence**: Brightness depends on the angle between the incoming light direction $\hat{v}_i$ and the surface normal $\hat{n}$.  
- **Energy conservation**: No light is reflected if the surface faces away from the light source.  

---

## Mathematical Model
The diffuse component is modeled using **Lambert’s Cosine Law**:

$$
L_d(\hat{v}_r; \lambda) = \sum_i L_i(\lambda) f_d(\lambda) [\hat{v}_i \cdot \hat{n}]_+
$$

where  

- $L_d(\hat{v}_r; \lambda)$ : Radiance due to diffuse reflection in direction $\hat{v}_r$  
- $L_i(\lambda)$ : Incident light intensity from source $i$  
- $f_d(\lambda)$ : Diffuse reflectance (wavelength-dependent material property) (think as constant BRD)
- $[\hat{v}_i \cdot \hat{n}]_+ = \max(0, \hat{v}_i \cdot \hat{n})$ : Accounts for self-shadowing  

---
## Visual Intuition
- Surfaces appear **brightest** when directly facing the light
  ($\theta_i = 0$).  
- Intensity **falls off** with the cosine of the incident angle.  
- At grazing angles, illumination goes to **zero**.  

Example analogy: A flashlight shining perpendicularly on a wall appears bright, but at an oblique angle, the same light covers more area and appears dimmer.

---
## Diagram
```
     Light Source
         ↓  (θi)
         \
          \     n (surface normal)
           \    ↑
            \   |
             \  |   Surface
              \ |___________________

Cosine falloff: Intensity ∝ cos(θi)
```

---
## Applications
- Core component of shading models (Phong, Blinn–Phong, Cook–Torrance, etc.)  
- Realistic rendering of matte materials (stone, clay, cloth, painted surfaces).  
- Basis for **diffuse albedo maps** in physically based rendering (PBR).  

---

## Links
- [[BRDF]]  
- [[Specular Reflection]]  
- [[Lambert's Cosine Law]]  
- [[Lighting Models]]  
