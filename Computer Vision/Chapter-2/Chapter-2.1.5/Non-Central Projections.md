# Non-Central Projection Models
 #cv/camera/distortion/non-central

Most traditional camera models assume a **central projection**: all image rays intersect at a single point (the **optical center**). 

This assumption holds for pinhole cameras and most standard lenses, where distortions can still be explained as deviations from a central projection (e.g., radial or tangential distortion).  

However, for some complex optical systems, this assumption breaks down. **Non-central projection models** are required when different image pixels do not share the same center of projection.

---

## 1. Motivation

### Central Projection Assumption
- A single optical center exists.  
- Rays from the scene pass through this point before reaching the sensor.  
- Enables modeling with simple perspective geometry:
  $$
  x_c = \frac{r_x \cdot p + t_x}{r_z \cdot p + t_z}, \quad 
  y_c = \frac{r_y \cdot p + t_y}{r_z \cdot p + t_z}
  $$

### Why Non-Central Models?
- Some lens systems **do not map all rays through one common center**.
- Examples include:
  - **Catadioptric cameras** (mirrors + lenses),
  - **Very wide-angle and fisheye systems**,
  - **Compound lenses with strong distortions**,
  - **Custom or experimental imaging devices**.

In these cases, standard central projection approximations produce large errors, making accurate calibration impossible without a non-central model.

---

## 2. Basic Concept

Instead of assuming a single optical center, **each pixel is associated with its own 3D projection line**.  

Formally:
- A pixel $(u, v)$ is mapped not to a single ray direction but to a **3D line**:
  $$
  L(u,v): \quad X(\lambda) = O(u,v) + \lambda D(u,v)
  $$
  where:
  - $O(u,v)$ = origin point of the projection line (not shared across all pixels),
  - $D(u,v)$ = direction vector,
  - $\lambda \in \mathbb{R}$ is a scalar along the line.

Thus, instead of one optical center, the camera is modeled as a **bundle of skew rays**.

---

## 3. Calibration

Calibrating a non-central model involves:
1. Capturing calibration images of known 3D patterns.
2. Estimating, for each pixel (or groups of pixels):
   - The local projection origin $O(u,v)$,
   - The projection direction $D(u,v)$.
3. Representing the set of rays either:
   - **Explicitly**: one line per pixel (dense but flexible),
   - **Parametrically**: using splines, polynomials, or basis functions for efficiency.

### Example Approaches
- **Spline-based models** (Goshtasby, 1989) approximate ray origins and directions smoothly across the image.
- **Ray-space models** (Grossberg & Nayar, 2001) represent the mapping between pixels and 3D rays as a **ray database**.

---

## 4. Applications

- **Omnidirectional vision**: Panoramic and 360° imaging systems.  
- **Catadioptric cameras**: Cameras using curved mirrors for wide field of view.  
- **Medical imaging optics**: Endoscopes or specialized optical fibers.  
- **Robotics and autonomous navigation**: Where accurate wide-angle geometry is crucial.  
- **Experimental optics research**, where custom imaging systems break central assumptions.

---

## 5. Advantages and Challenges

### Advantages
- Can accurately model **arbitrary distortions** where central models fail.
- Essential for systems without a single center of projection.
- Allows **sub-pixel geometric calibration** in complex optics.

### Challenges
- More complex calibration (requires more data and optimization).  
- Computationally heavier (each pixel may need its own mapping).  
- Harder to integrate into simple pinhole-based pipelines.  

---

## 6. Summary

- **Central models** assume a single optical center → sufficient for most cameras.  
- **Non-central models** treat each pixel as mapping to its own projection line → necessary for complex lenses, catadioptric systems, fisheye cameras, and non-standard optics.  
- Calibration typically relies on **spline or ray-space models** to capture these mappings.  

Non-central models expand the scope of computer vision to deal with unconventional optical systems, ensuring accurate reconstructions where simpler models fail.

---
**Next Pages**:  
- [[Radial Distortion]]  
- [[Tangential Distortion]]  
- [[Fisheye Distortion]]  
- [[Spline Distortion]]
