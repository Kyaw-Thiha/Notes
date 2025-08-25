# Di-chromatic Reflection Model

## Description
The **Di-chromatic Reflection Model** (Shafer, 1985) is based on the Torrance and Sparrow (1967) reflection framework. It states that the apparent color of a **uniform material** illuminated by a single light source can be expressed as the sum of **two reflection components**:

1. **Interface (specular) reflection** – reflection at the surface boundary.  
2. **Body (diffuse) reflection** – reflection due to subsurface scattering inside the material.  

This separation into **specular** and **body terms** allows the model to effectively explain the appearance of real-world materials.

---

## Formula
The outgoing radiance is written as:

$$
L_r(\hat{v}_r; \lambda) = L_i(\hat{v}_r, \hat{v}_i, \hat{n}; \lambda) + L_b(\hat{v}_r, \hat{v}_i, \hat{n}; \lambda)
$$

which expands into:

$$
L_r(\hat{v}_r; \lambda) = c_i(\lambda)m_i(\hat{v}_r, \hat{v}_i, \hat{n}) + c_b(\lambda)m_b(\hat{v}_r, \hat{v}_i, \hat{n})
$$

where  

- $c_i(\lambda)$ : Relative **power spectrum** of interface (specular) reflection  
- $m_i(\hat{v}_r, \hat{v}_i, \hat{n})$ : Geometric term for interface reflection  
- $c_b(\lambda)$ : Relative **power spectrum** of body (diffuse) reflection  
- $m_b(\hat{v}_r, \hat{v}_i, \hat{n})$ : Geometric term for body reflection  

Key property:  
- **Spectral terms** ($c(\lambda)$) depend **only on wavelength** (color information).  
- **Magnitude terms** ($m(\cdot)$) depend **only on geometry** (angles between light, surface, and viewer).  

---

## Relation to Phong Model
- Can be derived from a **generalized Phong model** assuming:  
  - A **single light source**.  
  - **No ambient illumination**.  
- Rearranging terms yields the separation into **interface** and **body** components.  

---

## Applications
The di-chromatic model has been widely used in **computer vision**:  

- **Specular–diffuse separation**: Segmenting colored objects with strong specular highlights and shading variations (Klinker, 1993).  
- **Color constancy**: Identifying intrinsic object color despite lighting conditions.  
- **Image processing**: Inspired local two-color models, e.g., for **Bayer pattern demosaicing** (Bennett et al., 2006).  

---

## Visual Intuition
- Specular reflection often preserves the **illumination color** (e.g., white from sunlight).  
- Body reflection contributes the **material’s intrinsic color** (e.g., red paint, green plastic).  
- The observed color is a **linear combination** of both.  

---

## Diagram (Conceptual)
```
         Incident Light (vi)
                ↓
        -------------------
       /     Surface       \
      /  [Interface term]   \
     /-----------------------\
    /   [Body reflection]     \
   /---------------------------\

Observed color = Specular + Diffuse
```

---

## Links
- [[Specular Reflection]]  
- [[Diffuse Reflection]]  
- [[Phong Shading Model]]  
- [[BRDF]]  
