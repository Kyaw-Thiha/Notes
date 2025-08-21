# Spotlight (Computer Vision / Graphics)
 #cv/light/spotlight #graphics

## Core Concept
- A **spotlight** is a type of point light source that emits light within a **restricted cone** of directions.  
- Defined by:  
  - **Position** (like a point light).  
  - **Direction** (axis of the cone).  
  - **Cutoff angle** (defines cone width).  
  - **Falloff function** (controls intensity fade inside the cone).  
- Used in CV and graphics for **controlled, localized illumination**.  

---

## Main Properties
- **Positioned source**: light originates from a specific point.  
- **Directional cone**: only illuminates within angle $\theta_c$.  
- **Falloff**: intensity decreases toward cone edges. Commonly modeled as:  

  $$
  I(\omega) = I_0 \cdot (\cos\phi)^n
  $$

  where:  
  - $I_0$: central intensity.  
  - $\phi$: angle between light direction and vector to surface point.  
  - $n$: falloff exponent (higher $n$ = sharper edges).  

- **Units**: flux per solid angle ($\text{W} \cdot \text{sr}^{-1}$).  

---
## Mathematical Formulation
- **Irradiance at point $p$ from spotlight at position $x_L$:**

  $$
  E(p) = \frac{I(\omega)}{r^2} \cos\theta
  $$

  - $r$: distance from $x_L$ to $p$.  
  - $\theta$: angle between incoming light direction and surface normal.  
  - $I(\omega)$: spotlight intensity in direction $\omega$.  
  - $I(\omega) = 0$ if $\phi > \theta_c$ (outside cone).  

---
## ASCII/LaTeX Diagram

```
         Spotlight (cone-shaped emission)
               \
                \
                 \   (cutoff θ_c)
                  \  v
                   \ | rays
                    \| 
                     * source position
                    / \
                   /   \
                  /     \
                 v       v
             Surface points illuminated
```

---
## Applications in CV
- **Photometric Stereo**: Spotlights are often used to create controlled directional lighting.  
- **Robotics & Vision Systems**: Focused beams illuminate regions of interest.  
- **Scene Relighting**: Used in AR/VR for localized lighting effects.  
- **Synthetic Data Generation**: Simulate flashlights, lamps, car headlights.  

---
## Limitations
- **Hard edges (ideal spotlight)**: Physically unrealistic unless falloff function used.  
- **Still a point source**: No area → shadows are sharp (no penumbra).  
- **Parameter sensitivity**: Position, cutoff, and falloff must be calibrated carefully in CV.  

---
## Key Takeaways
- Spotlight = **point light with a cone of influence**.  
- Adds **position, direction, cutoff, falloff** to the point light model.  
- Widely used in **controlled CV experiments**, **robotics lighting setups**, and **synthetic rendering**.  
