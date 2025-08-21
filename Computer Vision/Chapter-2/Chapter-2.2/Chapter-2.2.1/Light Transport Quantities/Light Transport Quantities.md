# Light Transport Quantities (Computer Vision)
#cv/light/radiometry

This page summarizes the **five fundamental light transport quantities** used in computer vision and graphics: **Radiance, Irradiance, Exitance, Intensity, Flux**.

---
## Radiance ($L$)
- **Definition**: Light energy per unit area per unit solid angle in a given direction.  
- **Units**: $\text{W} \cdot \text{sr}^{-1} \cdot \text{m}^{-2}$  
- **Key Idea**: Directional, conserved along rays.  
- **Role**: Fundamental in image formation and the rendering equation.  

[[Radiance|Read More]]

---
## Irradiance ($E$)
- **Definition**: Incoming flux per unit area on a surface.  
- **Units**: $\text{W} \cdot \text{m}^{-2}$  
- **Key Idea**: Sum of all incoming radiance, weighted by $\cos\theta$.  
- **Role**: Describes how much light arrives at a surface.  

[[Irradiance|Read More]]

---
## Exitance ($M$)
- **Definition**: Outgoing flux per unit area from a surface.  
- **Units**: $\text{W} \cdot \text{m}^{-2}$  
- **Key Idea**: Aggregate outgoing radiance (reflected/emitted).  
- **Role**: Describes how much light a surface contributes to the scene.  

[[Exitance|Read More]]

---
## Intensity ($I$)
- **Definition**: Flux per unit solid angle in a given direction.  
- **Units**: $\text{W} \cdot \text{sr}^{-1}$  
- **Key Idea**: Emission pattern of a point source.  
- **Role**: Models spotlights, isotropic lights, and other point emitters.  

[[Radiant Intensity|Read More]]

---
## Flux ($\Phi$)
- **Definition**: Total radiant power (energy per time).  
- **Units**: $\text{W}$  
- **Key Idea**: Integral measure, basis for all others.  
- **Role**: Describes total light output, input, or transfer.  

[[Radiance Flux|Read More]]

---
## Comparison Table

| Quantity       | Symbol | Units                                               | Definition                                    | Normalization              | Directional?              |
| -------------- | ------ | --------------------------------------------------- | --------------------------------------------- | -------------------------- | ------------------------- |
| **Flux**       | $\Phi$ | W                                                   | Total power                                   | None                       | ✗                         |
| **Radiance**   | $L$    | $\text{W} \cdot \text{sr}^{-1} \cdot \text{m}^{-2}$ | Power per area per solid angle in a direction | per area + per solid angle | ✔                         |
| **Irradiance** | $E$    | $\text{W} \cdot \text{m}^{-2}$                      | Incoming flux per unit area                   | per area                   | ✗ (hemisphere-integrated) |
| **Exitance**   | $M$    | $\text{W} \cdot \text{m}^{-2}$                      | Outgoing flux per unit area                   | per area                   | ✗ (hemisphere-integrated) |
| **Intensity**  | $I$    | $\text{W} \cdot \text{sr}^{-1}$                     | Flux per unit solid angle (point source)      | per solid angle            | ✔ (directional pattern)   |

---
## Key Takeaways
- **Flux ($\Phi$)**: total light power.  
- **Radiance ($L$)**: fundamental, directional, used in rendering equation.  
- **Irradiance ($E$)**: how much light lands on a surface.  
- **Exitance ($M$)**: how much light leaves a surface.  
- **Intensity ($I$)**: directional emission strength of a point source.  

Together, these form the **language of light transport in CV and graphics**.

--- 
## See More
- [[Radiance]]
- [[Irradiance]]
- [[Exitance]]
- [[Radiant Intensity]]
- [[Radiance Flux]]