# Photometric Image Formation

## (2.2.1) Lighting

### Light Transport Quantities
These are the **fundamental physical measures of light** used in computer vision and graphics.  

- **Radiance ($L$)** — light energy per unit area per unit solid angle in a direction (fundamental quantity).  
- **Irradiance ($E$)** — incoming flux per unit surface area (light arriving at a surface).  
- **Exitance ($M$)** — outgoing flux per unit surface area (light leaving a surface).  
- **Intensity ($I$)** — flux per unit solid angle (directional emission of a point source).  
- **Flux ($\Phi$)** — total radiant power (overall energy per time).  

[[Light Transport Quantities|Read More]]

---

### Light Models
These are the **idealized ways of representing light sources** in CV/graphics.  

- **Point Light** — single position emitter, $1/r^2$ falloff, sharp shadows.  
- **Area Light** — finite surface emitter, produces soft shadows.  
- **Directional Light** — parallel rays from infinity (e.g., sunlight), no falloff.  
- **Spotlight** — point light restricted to a cone, with cutoff + falloff.  
- **Environment Map** — illumination from all directions (sphere of radiance).  
- **HDR Environment Map** — environment map with high dynamic range values for accurate intensities.  
- **Sky Models** — analytic daylight distribution (depends on sun position + turbidity).  
- **Ambient Light** — uniform, directionless approximation of scattered light.  

[[Lighting Models|Read More]]
[[Lighting Models (Visual)|Visual Summary]]

---
## Reflectance & Shading
Reflectance & shading describe how surfaces **interact with incoming light** and how this interaction determines the **appearance of brightness and color** in images.  
- **BRDF (Bidirectional Reflectance Distribution Function)** — fundamental function linking incoming and outgoing light directions at a surface.  
- **Phong Shading Model** — empirical model combining ambient, diffuse, and specular reflection.  
- **Di-chromatic Reflection Model** — separates interface (specular) and body (diffuse) reflection for material analysis.  
- **Global Illumination** — accounts for multiple light bounces (indirect + direct light).  

[[Reflectance & Shading|Read More]]

---