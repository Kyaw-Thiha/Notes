# Radiant Intensity 
 #cv/light/radiometry/radiant-intensity

## Core Concept
- **Radiant Intensity** measures **power per unit solid angle** emitted by a point light source in a given direction.  
- Symbol: $I(\omega)$  
- Think of it as **“how strong the source looks in a particular direction”**, without considering area.  

---

## Main Properties
- **Units**: $\text{W} \cdot \text{sr}^{-1}$ (watts per steradian).  
- **Defined for point/approx. point sources**: Unlike radiance or irradiance, intensity is not per area.  
- **Directional**: Describes variation of emission pattern (e.g., spotlights vs isotropic lights).  

---

## Relation to Other Quantities
- From **Radiance ($L$)**:  

  $$
  I(\omega) = \int_A L(x, \omega) \cos\theta \, dA
  $$

  - $L(x,\omega)$: radiance at surface point $x$  
  - $A$: emitting surface area  
  - $\theta$: angle between $\omega$ and surface normal  

- To **Flux ($\Phi$)**:  

  $$
  \Phi = \int_{\Omega} I(\omega) \, d\omega
  $$

  where $\Phi$ is the total emitted power.  

---

## ASCII/LaTeX Diagram

```
     Radiant Intensity from a point source

          (ω₁) *
                \
                 \
                  \
                   -> Direction ω₁ (intensity I(ω₁))

          (ω₂) *
                \
                 \
                  \
                   -> Direction ω₂ (intensity I(ω₂))
```

LaTeX summary:

$$
I(\omega) = \frac{d\Phi}{d\omega}, 
\quad \Phi = \int_{\Omega} I(\omega) \, d\omega
$$

---

## Applications in CV
- **Lighting Models**: Describes emission patterns of point lights.  
- **Photometric Calibration**: Relating light source power to scene brightness.  
- **Inverse Rendering**: Estimating source intensity distribution from captured images.  
- **Spotlights vs Isotropic Lights**: Intensity explains why some directions appear brighter.  

---

## Relation to CV/Graphics Context
- **Radiance ($L$)**: per area, per solid angle → fundamental light transport quantity.  
- **Irradiance ($E$)**: power arriving at surface per unit area.  
- **Radiant Intensity ($I$)**: power per unit solid angle from source (no area term).  
- **Flux ($\Phi$)**: total emitted power.  

---

## Key Takeaways
- Radiant intensity = **“directional brightness of a source.”**  
- Converts between **radiance (at surfaces)** and **flux (total power)**.  
- Important for **modeling point light sources** in computer vision and graphics.  
