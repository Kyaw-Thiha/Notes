# Radiance 
 #cv/light/radiometry/radiance
## Core Concept
- **Radiance** is the measure of light energy traveling in a specific direction, per unit area, per unit solid angle.  
- In computer vision (CV), it connects **physical light transport** with how images are formed on a camera sensor.  
- Symbol: $L(x, \omega)$  
  - $x$: spatial position (on a surface or in space)  
  - $\omega$: direction of light  

---

## Main Properties
- **Units**: $\text{W} \cdot \text{sr}^{-1} \cdot \text{m}^{-2}$  
- **Conserved along rays**: Radiance does not diminish along a straight path in vacuum.  
- **Camera relevance**: Pixel intensity is proportional to the integral of radiance arriving through the lens aperture.

---

## Relation to Other Quantities
- **Irradiance ($E$)**: Total incoming light per unit area  

  $$
  E(x) = \int_{\Omega} L(x, \omega) \cos\theta \, d\omega
  $$

  where $\theta$ is the angle between direction $\omega$ and the surface normal.  

- **Radiant Intensity ($I$)**: Power per unit solid angle (from a point source).  

- **Flux ($\Phi$)**: Total power (integral of radiance over area + solid angle).  

---

## Radiance in Image Formation
- Camera pixel value ≈  

  $$
  \text{Pixel}(u,v) \propto \int_{\Omega} L(x, \omega) \, T(\omega) \, d\omega
  $$

  where $T(\omega)$ is lens transmission/response.  

- **Rendering Equation** (core of light transport):

  $$
  L_o(x, \omega_o) = L_e(x, \omega_o) + \int_{\Omega} f_r(x, \omega_i, \omega_o) \, L_i(x, \omega_i) \, \cos\theta \, d\omega_i
  $$

  - $L_o$: outgoing radiance  
  - $L_e$: emitted radiance  
  - $f_r$: BRDF (surface reflectance)  
  - $L_i$: incoming radiance  

---

## ASCII/LaTeX Diagram

### Intuition: Light Transport to Camera

```
   Light source
        *
       /|\
        |
        v   (incoming radiance L_i)
     ----------- surface (point x, normal n)
          \
           \  θ
            \      (outgoing radiance L_o(x, ω_o))
             \
              \
               \--> [Lens] --> [Sensor Pixel (u,v)]
```

LaTeX version:

$$
\text{Light Source} \;\; \longrightarrow \; L_i(x, \omega_i) 
\;\; \xrightarrow{\; f_r, \cos\theta \;} 
L_o(x, \omega_o) \;\; \longrightarrow \; \text{Camera Pixel}
$$

---

## Applications in CV
- **Photometric Stereo**: Uses radiance differences under multiple lighting conditions to recover surface normals.  
- **Inverse Rendering**: Estimate reflectance, illumination, geometry from captured radiance.  
- **Physically-Based Vision**: Bridges computer graphics models and real-world image formation.  
- **HDR Imaging**: Measures radiance more accurately by combining exposures.

---

## Key Takeaways
- Radiance is the **fundamental quantity of light transport** in CV.  
- Links **physics of light** with **camera imaging**.  
- Central to rendering equation, inverse rendering, and illumination models.  

---
## See Also
- [[Light Transport Quantities]]