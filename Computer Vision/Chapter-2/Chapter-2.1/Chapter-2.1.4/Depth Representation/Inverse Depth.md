# Inverse Depth
 #cv/camera/inverse-depth 

![Inverse vs Projective Depth](projective-inverse-depth.png)
**Inverse depth** is a way of representing the distance of a 3D point from the camera by taking the reciprocal of its Euclidean depth $z$ along the optical axis:

$$
d = \frac{1}{z}
$$

This representation is commonly used in computer vision, stereo reconstruction, and structure-from-motion because of its mathematical convenience and robustness.

---

## 1. Definition

- Let $p_w = (x_w, y_w, z_w)$ be a 3D point in **world coordinates**.  
- In the **camera coordinate system**, the depth of the point relative to the camera optical center is:
  $$
  z = r_z \cdot (p_w - c)
  $$
  where:
  - $r_z$: unit vector along the camera's $Z$ (optical) axis,  
  - $c$: camera center.  

- **Inverse depth** is then defined as:
  $$
  d = \frac{1}{z}
  $$
[[Computing Depth Along Camera Axis|Read here to see explanation]]

---

## 2. Properties

- **Nonlinear depth scale:**  
  Nearby points (small $z$) produce large inverse depth values; distant points (large $z$) compress toward zero.  
  - Example: $z=1 \to d=1$, $z=10 \to d=0.1$.

- **Stability:**  
  Inverse depth provides better numerical stability for representing very far-away points, as they naturally map close to zero.  
  This avoids large disparities in optimization.

- **Monotonic mapping:**  
  It preserves order: closer points always have larger inverse depth.

---

## 3. Relation to Disparity

In stereo vision:

- Disparity (pixel shift between left and right camera views) is **inversely proportional to depth**:
  $$
  \text{disparity} \propto \frac{1}{z}
  $$

Thus, **inverse depth is directly linked to disparity**, making it convenient for stereo and multi-view reconstruction.

---

## 4. Special Case of Projective Depth

Inverse depth is a **special case of projective depth (plane plus parallax)**:

$$
d = \frac{1}{z}
$$

This arises when the reference plane is chosen at infinity (i.e., $\hat{n}_0 = 0, \; c_0 = 1$).  
Thus, inverse depth can be viewed as projective depth relative to a plane at infinity.

---

## 5. Applications

- **Stereo vision:**  
  Directly relates to pixel disparity between camera views.  

- **Visual SLAM & structure-from-motion:**  
  Used to parameterize 3D points, enabling efficient bundle adjustment.  

- **Depth maps:**  
  Many depth estimation methods output inverse depth maps for stability and smoother gradients.  

- **Plane sweeping algorithms:**  
  Inverse depth layers are often used when sampling 3D space.

---

## 6. Summary

- Inverse depth: $d = 1/z$, a reciprocal representation of distance.  
- Compresses large depths, improving numerical stability.  
- Naturally tied to disparity in stereo vision.  
- Special case of projective depth with reference plane at infinity.  
- Widely used in **stereo reconstruction, SLAM, and multi-view geometry**.

## See Also
- [[Projective Depth]]
- [[Camera Matrix]]
- [[Computing Depth Along Camera Axis]]
