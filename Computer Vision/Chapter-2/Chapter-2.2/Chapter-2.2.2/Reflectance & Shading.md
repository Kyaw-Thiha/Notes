# Reflectance & Shading

## Description
**Reflectance** and **shading** describe how surfaces interact with light and how that interaction produces the colors and brightness we observe.  

- **Reflectance**: Material-dependent property describing how much incident light is reflected, transmitted, or absorbed.  
- **Shading**: The process of computing the outgoing radiance (brightness/color) for a given surface point under lighting and viewing conditions.  

Together, these concepts form the foundation of realistic rendering and computer vision.

---

## Core Concepts

### 1. BRDF (Bidirectional Reflectance Distribution Function)
- Defines how light scatters at a surface, relating **incoming** and **outgoing** directions.  
- General 4D function $f_r(\hat{v}_i, \hat{v}_r, \hat{n}; \lambda)$.  
- Splits into **diffuse** and **specular** components.  
- See: [[BRDF]]

---

### 2. Phong Shading Model
- Empirical model combining:  
  - **Ambient reflection** (global background illumination).  
  - **Diffuse reflection** (Lambertian cosine law).  
  - **Specular reflection** (Phong exponent for highlight sharpness).  
- Historically important, though replaced by microfacet models.  
- See: [[Phong Shading Model]]

---

### 3. Di-chromatic Reflection Model
- Separates reflection into:  
  - **Interface reflection** (specular, illumination color).  
  - **Body reflection** (diffuse, material color).  
- Useful in **computer vision** for color constancy and specular–diffuse separation.  
- See: [[Di-chromatic Reflection Model]]

---

### 4. Global Illumination
- Goes beyond local shading to include **indirect light bounces**.  
- Two classic methods:  
  - **Ray tracing / Path tracing**: Follows individual light rays (good for specular).  
  - **Radiosity**: Solves energy transfer between diffuse patches.  
- See: [[Global Illumination]]

---

## Additional Concepts

### Local vs. Global Models
- **Local models** (e.g., Phong, Lambertian): Consider only direct light from visible sources.  
- **Global models** (e.g., ray tracing, radiosity): Account for multiple bounces and interreflections.  

---

### Shading in Practice
- **Flat shading**: Single normal per polygon → faceted look.
- **Gouraud shading**: Interpolates vertex intensities → smoother surfaces.  
- **Phong shading (interpolation)**: Interpolates normals, computes shading per pixel.  

---

### Material Properties
- **Dielectrics**: Plastics, ceramics → diffuse color from body reflection, white specular.  
- **Metals**: Copper, gold → specular reflection inherits wavelength-dependent tint.  

---

### Common Effects
- **Shadows**: Occlusion of direct light.  
- **Color bleeding**: Surfaces tint neighboring surfaces (global illumination).  
- **Caustics**: Concentrated light patterns from specular/transmissive surfaces.  
- **Ambient occlusion**: Approximation of soft shadowing in concave regions.  

---
## Links
- [[BRDF]]  
- [[Phong Shading Model]]  
- [[Di-chromatic Reflection Model]]  
- [[Global Illumination]]  
- [[Diffuse Reflection]]  
- [[Specular Reflection]]  
- [[Lambert's Cosine Law]]  
- [[Chapter 2.2]]
