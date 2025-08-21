# Radiant Flux (Computer Vision / Light Transport)
 #cv/light/radiometry/radiant-flux

## Core Concept
- **Radiant Flux** (also called **Power**) is the **total energy per unit time** carried by light.  
- Symbol: $\Phi$  
- It is the most **global measure of light transport**: unlike radiance, irradiance, or intensity, it has no per-area or per-angle normalization.  

---

## Main Properties
- **Units**: $\text{W}$ (watts).  
- **Represents total power**: Integrates over space, area, or solid angle depending on context.  
- **Broadest measure**: Other light transport quantities (radiance, irradiance, intensity) are derivatives of flux.  

---

## Relation to Other Quantities
- **From Radiance ($L$):**  

  $$
  \Phi = \int_A \int_{\Omega} L(x, \omega) \cos\theta \, d\omega \, dA
  $$

  where:  
  - $A$: surface area  
  - $\Omega$: solid angle of directions  
  - $\theta$: angle between $\omega$ and surface normal  

- **From Intensity ($I$):**  

  $$
  \Phi = \int_{\Omega} I(\omega) \, d\omega
  $$

- **From Irradiance ($E$):**  

  $$
  \Phi = \int_A E(x) \, dA
  $$

---

## ASCII/LaTeX Diagram

```
    Radiant Flux (Φ) = All Light Energy per Time

    Source ---> [ Light rays in many directions ]
             ↘   ↓   ↙
              ↘  ↓  ↙
               ↘ ↓ ↙
               [ Surface / Scene / Sensor ]
    
    Summed over all rays, areas, and angles = Φ
```

LaTeX summary:

$$
\Phi = \frac{dQ}{dt}
\quad\;\; \text{where $Q$ is radiant energy (Joules).}
$$

---

## Applications in CV
- **Light Source Power**: Defines total output of a lamp or LED.  
- **Camera Exposure Models**: Total flux reaching a sensor influences brightness.  
- **Photometry & Calibration**: Flux helps bridge between physical light measurements and pixel values.  
- **Energy Conservation in Rendering**: Ensures total emitted = total absorbed + reflected + transmitted.  

---

## Relation to Other Light Quantities
- **Radiance ($L$):** direction + area resolved (fundamental transport quantity).  
- **Irradiance ($E$):** flux per unit area (incoming light density).  
- **Radiant Intensity ($I$):** flux per unit solid angle (directional emission).  
- **Flux ($\Phi$):** total power (integral over all).  

---
## Key Takeaways
- Radiant Flux = **“total light power.”**  
- Other light transport measures are derived from it by normalizing per area or per solid angle.  
- Crucial in **CV calibration**, **illumination modeling**, and **energy-based rendering equations**.  

---
## See Also
- [[Light Transport Quantities]]