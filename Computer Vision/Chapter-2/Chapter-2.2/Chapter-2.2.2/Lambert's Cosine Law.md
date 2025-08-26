# Lambert's Cosine Law

## Description
**Lambert's Cosine Law** (also called **Lambert’s Cosine Formula**) describes how the **apparent brightness** of a diffusely reflecting surface depends on the angle of incident light relative to the surface normal.  

It is the foundation of **diffuse reflection** modeling in computer graphics and physics. The law states that the observed intensity is proportional to the cosine of the angle between the surface normal and the incoming light direction.

---

## Formula
For a surface illuminated by incident light $\hat{v}_i$ with surface normal $\hat{n}$:

$$
I \propto \cos \theta_i = \hat{v}_i \cdot \hat{n}
$$

where  

- $I$ : Observed intensity of reflected light  
- $\theta_i$ : Angle between incident light direction $\hat{v}_i$ and surface normal $\hat{n}$  
- $\hat{v}_i \cdot \hat{n}$ : Dot product, capturing cosine dependence  

To ensure non-negative values, the convention is:

$$
[\hat{v}_i \cdot \hat{n}]_+ = \max(0, \hat{v}_i \cdot \hat{n})
$$

---

## Shading Equation (Diffuse Case)
Lambert’s law is directly used in the diffuse reflection model:

$$
L_d(\hat{v}_r; \lambda) = \sum_i L_i(\lambda) f_d(\lambda) [\hat{v}_i \cdot \hat{n}]_+
$$

where  

- $L_d$ : Reflected radiance in direction $\hat{v}_r$  
- $L_i(\lambda)$ : Incident radiance from source $i$  
- $f_d(\lambda)$ : Diffuse reflectance (material property)  

---

## Visual Intuition
- Surface looks **brightest** when directly facing the light ($\theta_i = 0$).  
- Brightness **falls off smoothly** with $\cos(\theta_i)$.  
- At grazing angles ($\theta_i \to 90^\circ$), intensity approaches **zero**.  

Example: A flashlight shining perpendicularly onto a wall appears bright, but as the angle increases, the same light spreads across a larger area and looks dimmer.

---

## Diagram
```
     Light source
         ↓
         \
          \   θi
           \    \
            \    \   n (normal)
             \    ↑
              \   |
               \  |
                \ |_______ Surface

Apparent intensity ∝ cos(θi)
```

---

## Applications
- Fundamental to **diffuse reflection** (Lambertian model).  
- Basis for simple shading in computer graphics (Phong, Blinn–Phong).  
- Used in **radiometry** and **photometry** to model energy distribution.  

---

## Links
- [[Diffuse Reflection]]  
- [[BRDF]]  
- [[Specular Reflection]]  
- [[Reflectance & Shading]]
- [[Lighting Models]]  
- [[Radiant Intensity]]
