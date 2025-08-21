# Exitance 
 #cv/light/radiometry/exitance

## Core Concept
- **Exitance** (also called **Radiant Exitance**) is the measure of **outgoing flux per unit surface area** from a surface.  
- Symbol: $M(x)$  
- Represents how much total light power (reflected, transmitted, or emitted) leaves a surface.  

---

## Main Properties
- **Units**: $\text{W} \cdot \text{m}^{-2}$  
- **Aggregate measure**: Unlike radiance (directional), exitance sums over all outgoing directions.  
- **Surface-based**: Defined at a point or over an area of a surface.  

---

## Relation to Other Quantities
- **From Radiance ($L$):**

  $$
  M(x) = \int_{\Omega} L(x, \omega) \cos\theta \, d\omega
  $$

  - $\Omega$: hemisphere of outgoing directions  
  - $\theta$: angle between direction $\omega$ and surface normal  

- **From Flux ($\Phi$):**

  $$
  M = \frac{d\Phi}{dA}
  $$

  i.e. flux per unit surface area.  

- **Comparison with Irradiance ($E$):**  
  - Irradiance = incoming flux density  
  - Exitance = outgoing flux density  

---

## ASCII/LaTeX Diagram

```
         Outgoing Radiance in all directions
            ↗   ↑   ↖
             ↗  ↑  ↖
              ↗ ↑ ↖
          ----------------
          Surface (point x, normal n)

Exitance M(x) = total outgoing flux / area
```

LaTeX summary:

$$
M(x) = \int_{\Omega} L(x, \omega) \cos\theta \, d\omega
\quad\;\; \text{(outgoing)}
$$

---

## Applications in CV
- **Shading & Rendering**: Used in diffuse reflection models (Lambertian surfaces emit uniformly over the hemisphere).  
- **Inverse Rendering**: Helps estimate reflectance/albedo from observed outgoing light.  
- **Global Illumination**: Tracks how much energy leaves surfaces to contribute to other surfaces’ irradiance.  
- **Material Appearance**: Captures overall brightness a surface contributes to the scene.  

---

## Relation to Light Quantities
- **Flux ($\Phi$):** total power.  
- **Radiance ($L$):** directional, per area per solid angle.  
- **Irradiance ($E$):** incoming flux per unit area.  
- **Exitance ($M$):** outgoing flux per unit area.  
- **Intensity ($I$):** flux per unit solid angle (for sources).  

---

## Key Takeaways
- Exitance = **“how much light leaves a surface per unit area.”**  
- Complements irradiance (incoming).  
- Essential for shading models and energy conservation in rendering equations.  

---
## See Also
- [[Light Transport Quantities]]