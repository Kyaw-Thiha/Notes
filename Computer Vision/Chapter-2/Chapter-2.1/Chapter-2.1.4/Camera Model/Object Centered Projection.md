# Object-Centered Projection
 #cv/camera/object-centered-projection

Object-centered projection is a reformulation of perspective projection that helps decouple **focal length** and **object distance**, which are otherwise strongly correlated.
This model is especially useful when working with **long focal length (telephoto) lenses**, where estimating focal length from image measurements alone becomes unreliable.

---

## 1. The Ambiguity of Focal Length vs. Distance

- In a standard perspective projection, the apparent **scale of an object** depends on the ratio:

$$
\text{scale} \propto \frac{f}{t_z}
$$

where:  
- $f$: focal length  
- $t_z$: distance from the camera to the object center  

This creates an **ambiguity**:
- A zoom-in (increase in $f$) has the same effect as moving closer (decrease in $t_z$).  
- This is dramatically illustrated in Hitchcock’s *Vertigo* (the "dolly zoom" effect).

---

## 2. Projection Equations

Using a simple [[Camera Matrix|calibration matrix]] $K$ (Eq. 2.59), the perspective projection can be written as:

$$
x_s = \frac{f \, (r_x \cdot p + t_x)}{r_z \cdot p + t_z} + c_x
\tag{2.73}
$$

$$
y_s = \frac{f \, (r_y \cdot p + t_y)}{r_z \cdot p + t_z} + c_y
\tag{2.74}
$$

where:  
- $p = (X, Y, Z)$: 3D point in object-centered coordinates.  
- $r_x, r_y, r_z$: rows of the rotation matrix $R$.  
- $(t_x, t_y, t_z)$: translation components.  
- $(c_x, c_y)$: principal point in pixel coordinates.  

If $t_z \gg \|p\|$ (object small compared to distance), the denominator is approximately $t_z$, reinforcing that scale depends on $f/t_z$.

---

## 3. Object-Centered Reparameterization

To decouple focal length and distance:

- Define **inverse distance**:  
  $$
  \eta_z = \frac{1}{t_z}
  $$

- Define **scale parameter**:  
  $$
  s = \eta_z f
  $$

Rewritten projection equations:

$$
x_s = \frac{s (r_x \cdot p + t_x)}{1 + \eta_z (r_z \cdot p)} + c_x
\tag{2.75}
$$

$$
y_s = \frac{s (r_y \cdot p + t_y)}{1 + \eta_z (r_z \cdot p)} + c_y
\tag{2.76}
$$

---

## 4. Advantages of Object-Centered Projection

- **Scale $s$** can be estimated reliably if the 3D object geometry $p$ is known.  
- **Inverse depth $\eta_z$** is then decoupled from scale and can be estimated from **foreshortening** (changes in shape as the object rotates).  
- As focal length increases (telephoto lens), $\eta_z \to 0$ and the model naturally approaches **orthographic projection**:
  - No need to switch to a different projection model.  
  - Provides a smooth link between **orthographic factorization methods** and **projective (perspective) methods**.  

---

## 5. Summary

- Standard perspective projection mixes focal length and distance, making them hard to separate.  
- Object-centered projection reparameterizes the model with $(s, \eta_z)$:  
  - $s$: overall scale (focal length and distance combined).  
  - $\eta_z$: inverse distance, tied to foreshortening.  
- This formulation is especially useful for:
  - **Long focal length imaging**.  
  - **3D reconstruction** where orthographic and perspective models need to be unified.  

## See Also
- [[Camera Matrix]]
- [[Focal Length]]
- [[Projective Depth]]
- [[Para-Perspective Projection]]
- [[Perspective Projection]]