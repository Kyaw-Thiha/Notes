# Global Illumination (Ray Tracing and Radiosity)

## Description
**Global illumination** refers to rendering methods that account for **indirect lighting** effects, where light rays bounce multiple times in a scene before reaching the camera. Unlike simple shading models (which consider only direct illumination), global illumination captures realistic phenomena such as:  
- Shadows cast by occluders.  
- Multiple bounces of light between surfaces.  
- Color bleeding and caustics (e.g., rippling water effects).  

Two primary classical techniques are used: **Ray Tracing (Path Tracing)** and **Radiosity**.  

---

## Ray Tracing (Path Tracing)

### Idea
Ray tracing simulates light transport by following individual rays across multiple bounces. It is best suited for **specular scenes** (glass, mirrors, polished objects).  

### Algorithm
1. Cast a primary ray from the **camera** through each pixel.  
2. Find the **nearest intersection** with scene geometry.  
3. Compute **direct lighting** at the intersection (Phong or BRDF models).  
4. Cast **secondary rays**:  
   - **Specular reflection rays** → mirror/glossy surfaces.  
   - **Transmission/refraction rays** → transparent materials.  
   - **Shadow rays** → determine visibility of light sources.  
5. Accumulate contributions from multiple bounces, including color attenuation.  

### Variants
- **Shadow mapping / shadow buffers**: Alternative to secondary rays for computing illumination visibility (Williams, 1983).  
- **Path tracing**: Follows rays stochastically to approximate global illumination with Monte Carlo integration.  

---
## Radiosity

### Idea
Radiosity is best suited for **diffuse scenes** (uniform albedo, matte surfaces). It models light exchange between surface patches using energy conservation principles.  

### Method
- Divide the scene into **rectangular patches** (including area light sources).  
- Compute **form factors**: amount of light transferred between patch $i$ and $j$, depending on:  
  - Orientation of patches.  
  - Reflectance properties.  
  - $1/r^2$ fall-off with distance.  
- Set up a **large linear system** of equations:  

$$
B_i = E_i + \rho_i \sum_j F_{ij} B_j
$$  

where  
- $B_i$ = Radiosity (light leaving patch $i$)  
- $E_i$ = Emission from patch $i$ (light sources)  
- $\rho_i$ = Reflectance of patch $i$  
- $F_{ij}$ = Form factor between patches $i$ and $j$  

- Solve the system to obtain patch illumination.  
- Render from any viewpoint using the solved radiosities.  

### Limitations
- Does not handle near-field effects well (e.g., **corner darkening**, **limited ambient illumination**).  
- Ignores subtle inter-reflection details without extensions.  

---
## Comparison
| Aspect               | Ray Tracing (Path Tracing) | Radiosity |
|----------------------|-----------------------------|-----------|
| Best for             | Specular, reflective, refractive scenes | Diffuse, matte scenes |
| Method               | Traces rays per pixel       | Solves linear system of patch-to-patch exchanges |
| Handles reflections? | Yes (mirror, refraction, caustics) | No (only diffuse) |
| Handles shadows?     | Yes (via secondary rays or shadow maps) | Yes (via form factors) |
| Efficiency           | Expensive per pixel, scalable with GPUs | Expensive global solve, viewpoint independent |
| Output use           | Photo-realistic rendering   | Architectural lighting simulation |

---
## Applications
- **Ray tracing / path tracing**: Glass, metals, caustics, complex reflections, realistic movies and games.  
- **Radiosity**: Architectural visualization, diffuse environment rendering.  
- **Hybrid methods**: Combine radiosity for diffuse effects and ray tracing for specular reflections (Wallace et al., 1987).  
- **Computer vision**: Estimating global illumination from real images (Yu et al., 1999).  

---
## Links
- [[BRDF]]  
- [[Phong Shading Model]]  
- [[Diffuse Reflection]]  
- [[Specular Reflection]]  
