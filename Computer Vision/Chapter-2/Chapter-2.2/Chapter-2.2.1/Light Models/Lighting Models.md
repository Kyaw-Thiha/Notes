# Lighting Models (Computer Vision / Graphics)
#cv/light #graphics 

This page summarizes the **major lighting models** used in computer vision (CV) and computer graphics.  
Each section links to a more detailed note.  

---

## Point Light Source
- Emits light from a **single point in space**.  
- Has **position**, but no size or shape.  
- Follows **inverse-square falloff**: $E \propto \frac{1}{r^2} \cos\theta$.  
- ✅ Useful for: simple CV setups, flashlights, lamps.  
- 🔗 See: [[Point Light Source]]  

---

## Area Light Source
- Emits light from a **finite surface (rectangle, disk, sphere, etc.)**.  
- Produces **soft shadows** due to finite size.  
- Requires integration across the surface.  
- ✅ Useful for: realistic rendering, light panels in CV experiments.  
- 🔗 See: [[Area Light]]  

---

## Directional Light
- Models light from an **infinitely distant source** (all rays parallel).  
- No falloff with distance.  
- Example: **sunlight**.  
- ✅ Useful for: outdoor CV, shadow studies, diffuse shading.  
- 🔗 See: [[Directional Light]]  

---

## Spotlight
- A **point light with a cone of emission**.  
- Defined by **position, direction, cutoff angle, falloff exponent**.  
- Produces localized, directional illumination.  
- ✅ Useful for: robotics, flashlights, focused illumination.  
- 🔗 See: [[Spotlight]]  

---

## Environment Map
- Represents illumination as a **function over directions** (spherical panorama).  
- Assumes **infinite distance sources**.  
- Typically stored as cube map or lat-long image.  
- ✅ Useful for: relighting, AR/VR, IBL.  
- 🔗 See: [[Environment Map]]  

---

## HDR Environment Map
- Environment map stored in **high dynamic range (HDR)** format.  
- Preserves accurate radiance values across wide intensity ranges.  
- Captures sunlight + shadows + environment variation realistically.  
- ✅ Useful for: physically-based rendering, inverse rendering, relighting.  
- 🔗 See: [[HDR Environment Map]]  

---

## Skylight / Sky Models
- Analytical models of **daylight distribution** across sky dome.  
- Inputs: sun position, turbidity (atmospheric clarity).  
- Examples: **Preetham**, **Hosek-Wilkie**, **CIE Standard skies**.  
- ✅ Useful for: outdoor CV, autonomous driving, scene relighting.  
- 🔗 See: [[Sky Models]]  

---

## Ambient Light
- Simplified **uniform illumination** from all directions.  
- Computationally cheap, but not physically accurate.  
- Ignores shadows and variation.  
- ✅ Useful for: baseline models, fallback assumptions.  
- 🔗 See: [[Ambient Light]]  

---

# Comparison Table

| Model              | Position | Direction | Size/Extent | Falloff         | Shadows | Typical Use Case |
|--------------------|----------|-----------|-------------|-----------------|---------|------------------|
| **Point Light**    | ✔        | ✗         | ✗           | $1/r^2$         | Sharp   | Simple lamps, flashes |
| **Area Light**     | ✔        | ✔ (per point) | ✔ (finite) | $1/r^2$ (per emitter point) | Soft    | Panels, windows |
| **Directional**    | ✗        | ✔         | ✗ (infinite source) | None (constant) | Sharp   | Sunlight, outdoor CV |
| **Spotlight**      | ✔        | ✔ (cone)  | ✗           | $1/r^2$ + cone falloff | Sharp | Flashlight, robotics |
| **Environment Map** | ✗        | ✔ (all directions) | Whole sphere | None (distant) | Mixed   | Relighting, AR/VR |
| **HDR Env. Map**   | ✗        | ✔         | Whole sphere | None (distant) | Mixed   | Physically-accurate IBL |
| **Sky Model**      | ✗        | ✔ (continuous) | Whole dome | None (distant) | Mixed   | Outdoor daylight |
| **Ambient Light**  | ✗        | ✗         | Global       | None            | None    | Approximation / baseline |

---

## Key Takeaways
- **Local lights (Point, Area, Spot)** → good for controlled setups and indoor CV.  
- **Distant lights (Directional, Sky, Environment maps)** → essential for outdoor scenes and global illumination.  
- **Ambient light** → useful as a simple approximation but not physically accurate.  
- **HDR maps** → required for realism and physically-based CV/graphics applications.  

[[Lighting Models (Visual)|Visual Summary]]

---
## See Also
- [[Point Light Source]]
- [[Area Light]]
- [[Directional Light]]
- [[Spotlight]]
- [[Environment Map]]
- [[HDR Environment Map]]
- [[Sky Models]]
- [[Ambient Light]]
- [[Lighting Models (Visual)]]
- [[Chapter 2.2]]