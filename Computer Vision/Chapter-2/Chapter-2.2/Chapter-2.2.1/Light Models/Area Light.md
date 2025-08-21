# Area Light Source (Computer Vision / Graphics)
 #cv/light/area #graphics

## Core Concept
- An **area light source** is a finite surface that emits light across its entire area.  
- Unlike **point lights**, area lights have **spatial extent**, leading to **soft shadows** and more realistic illumination.  
- Common in real-world CV setups, since most lights (lamps, LEDs, windows) occupy an area.  

---

## Main Properties
- **Defined by geometry**: position, shape (e.g., rectangle, disk, sphere), and orientation.  
- **Emission distribution**:  
  - *Lambertian emitter*: emits uniformly in all directions from each surface point.  
  - *Directional emitters*: emission depends on angle.  
- **Finite size**: leads to smooth variation in illumination.  

---

## Mathematical Formulation
- **Radiant Exitance ($M$)** from surface element $dA$:  

  $$
  d\Phi = L(x, \omega) \cos\theta \, d\omega \, dA
  $$

- **Irradiance at point $p$** from an area light $A$:  

  $$
  E(p) = \int_A \frac{L(x, \omega) \cos\theta_i \cos\theta_p}{r^2} \, dA
  $$

  where:  
  - $L(x, \omega)$: radiance from source point $x$ in direction $\omega$  
  - $r$: distance between $x$ (light point) and $p$ (receiver)  
  - $\theta_i$: angle at light (between normal and direction to $p$)  
  - $\theta_p$: angle at receiver surface (between normal and incoming ray)  

- **Key Difference from Point Light**: integral over area $A$, not just one position.  

---

## ASCII/LaTeX Diagram

```
         Area Light Source (rectangle)
         -----------------------------
        |                             |
        |                             |
        |                             |
         -----------------------------
            \    \   |   /    /
             \    \  |  /    /
              \    \ | /    /
               v    v v    v
              Surface point (p)

Multiple rays from different parts of light
→ soft shadow edges (penumbra)
```

---

## Applications in CV
- **Realistic Illumination Models**: Most real-world lighting is extended, not point-like.  
- **Soft Shadow Estimation**: Important in shape-from-shading and photometric analysis.  
- **Inverse Rendering**: More accurate scene relighting by modeling finite light geometry.  
- **Light Stage Systems**: In CV research, LED panels approximate area lights for uniform illumination.  

---

## Limitations
- **Computation-heavy**: Requires integration across area (Monte Carlo sampling in rendering).  
- **Harder to calibrate**: Exact geometry and emission profile must be known for precision.  

---

## Key Takeaways
- Area lights = **finite surface emitters**, more realistic than point lights.  
- Produce **soft shadows** and smoother illumination.  
- Widely used in CV and graphics for modeling real-world lighting conditions.  

---
## See Also
- [[Radiant Intensity]]
- [[Radiance]]
- [[Irradiance]]
