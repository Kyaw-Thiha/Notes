# Irradiance 
 #cv/light/radiometry/irradiance

## Core Concept
- **Irradiance** measures the total **incoming light power per unit area** on a surface.  
- Symbol: $E(x)$  
- Unlike **radiance** (which tracks directional light), irradiance **integrates light from all incoming directions** over the hemisphere above a surface.  

---

## Main Properties
- **Units**: $ \text{W} \cdot \text{m}^{-2} $  
- **Direction-agnostic**: Aggregates incoming radiance regardless of direction.  
- **Surface-based**: Defined at a point on a surface, influenced by orientation (normal).  

---

## Relation to Radiance
- Irradiance is obtained by integrating radiance with a cosine weighting:  

  $$
  E(x) = \int_{\Omega} L(x, \omega) \cos\theta \, d\omega
  $$

  - $L(x, \omega)$: incoming radiance at point $x$ from direction $\omega$  
  - $\theta$: angle between direction $\omega$ and surface normal  
  - $\Omega$: hemisphere of incoming directions  

- **Key difference**:  
  - Radiance $L(x, \omega)$ → direction-specific  
  - Irradiance $E(x)$ → total incident light power density  

---

## ASCII/LaTeX Diagram

```
     Incoming Radiance from many directions
         ↘    ↓    ↙
          ↘   ↓   ↙
           ↘  ↓  ↙
            ↘ ↓ ↙
         -----------
          Surface (point x, normal n)

Irradiance = sum of all incoming radiance
             weighted by cos(θ)
```

LaTeX version:

$$
E(x) = \int_{\Omega} L(x, \omega) \cos\theta \, d\omega
$$

---

## Applications in CV
- **Shading Models**: Irradiance determines how much light reaches a point before reflection (used in Lambertian shading).  
- **Photometric Stereo**: Irradiance variations under different lights help recover surface normals.  
- **Illumination Estimation**: Capturing scene brightness distribution from light sources.  
- **Global Illumination**: Used in rendering equations for diffuse reflection components.  

---

## Relation to Other Quantities
- **Radiance ($L$)**: directional measure of light.  
- **Irradiance ($E$)**: total incoming radiance per area.  
- **Exitance ($M$)**: outgoing flux per unit area (after reflection or emission).  
- **Flux ($\Phi$)**: total power, integral of irradiance over surface.  

---

## Key Takeaways
- Irradiance = **“How much light lands on a surface?”**  
- Derived from radiance via cosine-weighted integration.  
- Central to image formation, shading, and illumination models.  
