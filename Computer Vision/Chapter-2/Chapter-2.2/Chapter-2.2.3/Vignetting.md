# Vignetting  
#cv/optics/vignetting 

![[vignetting.png]]
## Core Concept  
- **Vignetting** is the falloff of brightness toward the edges of an image.  
- It arises from two main causes:  
  - **Natural vignetting**: due to geometry of light propagation and lens aperture.  
  - **Mechanical vignetting**: due to occlusion of rays inside the lens design.  

---

## Natural Vignetting (cos⁴ Law)  

### Step-by-Step Derivation (from Szeliski, 2.2.3)  
1. **Foreshortening of object patch**  
   - Light from object patch $\delta_o$ at off-axis angle $\alpha$ is reduced by $\cos \alpha$.  

2. **Distance effect**  
   - Effective distance $r_o = \tfrac{z_o}{\cos\alpha}$.  
   - Light falls off as $1/r_o^2$.  

3. **Foreshortened aperture**  
   - Aperture ellipse has area $\pi (d/2)(d \cos\alpha/2) = \tfrac{\pi}{4} d^2 \cos\alpha$.  

4. **Combined effect**  
   - Amount of light through aperture:  

     $$
     \delta_o \cdot \frac{\cos\alpha}{r_o^2} \cdot \frac{\pi}{4} d^2 \cos\alpha
     = \delta_o \cdot \frac{\pi}{4} \frac{d^2}{z_o^2} \cos^4\alpha
     \tag{2.99}
     $$  

5. **Relating object area to pixel area**  
   - From similar triangles:  

     $$
     \frac{\delta_o}{\delta_i} = \frac{z_o^2}{z_i^2}
     \tag{2.100}
     $$  

6. **Final irradiance relation**  
   - Substituting into (2.99):  

     $$
     \delta_i \cdot \frac{\pi}{4} \frac{d^2}{z_i^2} \cos^4\alpha
     \approx \delta_i \cdot \frac{\pi}{4} \left(\frac{d}{f}\right)^2 \cos^4\alpha
     \tag{2.101}
     $$  

   - Linking **scene radiance $L$** to **pixel irradiance $E$**:  

     $$
     E = L \cdot \frac{\pi}{4} \left(\frac{d}{f}\right)^2 \cos^4 \alpha
     \tag{2.102}
     $$  

---

## Mechanical Vignetting  
- Caused by **internal occlusion of rays** near the edges of the lens.  
- Cannot be captured with simple formulas → requires **ray tracing of actual lens geometry**.  
- Strongest at **wide apertures (low f-number)**.  
- Reduced by stopping down (increasing f-number).  

---

## ASCII/LaTeX Diagram  

### Natural Vignetting Geometry  
```
Object patch δo (off-axis, angle α)
       \
        \        Aperture (ellipse d × d cosα)
         \     /------O------\
          \   |               |
           \--|     Lens      |---> Pixel δi (dimmer)
              \             /
               \-----------/
```

Brightness $\propto \cos^4 \alpha$ at the pixel.  

---

## Implications  
- Smaller sensors → less light per pixel → noisier images.  
- Brightness depends on:  
  - Aperture size $d$ (or equivalently f-stop $N = f/d$).  
  - Off-axis angle $\alpha$ (cos⁴ law).  
- Explains why **edges of wide-angle photos appear darker**.  

---

## Calibration & Correction  
- Measured via:  
  - **Integrating spheres**.  
  - **Uniform targets**.  
  - **Camera rotation**.  
- Corrected in software by applying **inverse gain map**.  

---

## Applications in CV  
- **Photometric stereo**: uncorrected vignetting biases surface reflectance estimates.  
- **Panorama stitching**: uneven brightness across overlapping images must be normalized.  
- **Rendering**: adding vignetting increases realism.  

---

## Key Takeaways  
- Natural vignetting follows the **cos⁴ law** (Eq. 2.102).  
- Mechanical vignetting comes from physical lens occlusion.  
- Both can be corrected, but natural vignetting has a neat closed-form formula.  

---

## See Also  
- [[Optics]]  
- [[Chromatic Aberration]]  
- [[Radiance]]  
- [[Irradiance]]  

