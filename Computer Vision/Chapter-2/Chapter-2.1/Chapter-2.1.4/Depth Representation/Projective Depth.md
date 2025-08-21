# Projective Depth (Plane plus Parallax)
 #cv/camera/projective-depth

![Projective vs Inverse Depth](projective-inverse-depth.png)
**Projective depth** (also called **parallax** or **plane plus parallax**) is a generalization of the standard notion of disparity (inverse depth). 
It arises when using the extended $4 \times 4$ camera matrix $\tilde{P}$, which provides extra flexibility in how we represent and sample 3D space for multi-view reconstruction.

---
## 1. Standard Disparity vs. Projective Depth

- **Standard disparity (inverse depth):**
  $$
  d = \frac{1}{z}
  $$
  where $z$ is the distance of a 3D point from the camera along the optical axis.

- **Projective depth:**  
  Generalizes disparity by allowing the last row of $\tilde{P}$ to be **remapped arbitrarily**, introducing the concept of depth relative to a **reference plane**.

---
## 2. General Formulation

Let the last row of $\tilde{P}$ be defined as:

$$
p_3 = s_3 \, [\hat{n}_0 \;|\; c_0]
$$

where:

- $\hat{n}_0$: unit normal vector of a chosen reference plane ($\|\hat{n}_0\| = 1$).  
- $c_0$: offset term.  
- $s_3$: scaling constant.

Then the projective depth $d$ of a 3D point $p_w$ is:

$$
d = \frac{s_3}{z} \, (\hat{n}_0 \cdot p_w + c_0)
\tag{2.66}
$$

with:

- $z = r_z \cdot (p_w - c)$  
  = distance from $p_w$ to the camera center $C$ along the **optical axis** $Z$.  

Thus, **$d$ measures disparity relative to the reference plane**:

$$
\hat{n}_0 \cdot p_w + c_0 = 0
$$

[[Understanding Projective Depth|Read here for detailed explanation]]

---
## 3. Special Case: Reference Plane at Infinity

If we set:

- $\hat{n}_0 = 0$  
- $c_0 = 1$

then the reference plane is at infinity, and the formula reduces to the **standard disparity model**:

$$
d = \frac{1}{z}
$$

---

## 4. Interpretation

- **Geometric meaning:**  
  $d$ encodes how far a point lies from the chosen **reference plane**, scaled by its optical-axis distance from the camera.  
- **Alternative name:** In many reconstruction algorithms, $d$ is called **parallax**.  
- **Plane plus parallax:** This framework models scene geometry as a **reference plane** plus **parallax displacements** of points relative to it.

---
## 5. Inverting the Mapping

The $4 \times 4$ camera matrix $\tilde{P}$ can be inverted to map pixels plus disparity back to 3D world coordinates:

$$
\tilde{p}_w = \tilde{P}^{-1} x_s
\tag{2.67}
$$

This allows direct recovery of 3D points from $(x_s, y_s, d)$ triplets.

---
## 6. Applications

- **Multi-view stereo reconstruction:**  
  By choosing $\tilde{P}$ flexibly, we can sweep a set of reference planes through space, each defining a projective depth sampling.  
- **Plane-sweeping algorithms:**  
  Efficiently match and reconstruct 3D surfaces by testing multiple candidate planes.  
- **Variable sampling:**  
  Projective depth allows **non-uniform depth sampling** better matched to observed parallax, improving efficiency and accuracy.

---
## 7. Summary

- **Projective depth** generalizes inverse depth by referencing disparity to an arbitrary plane.  
- Formula:  
  $$
  d = \frac{s_3}{z} (\hat{n}_0 \cdot p_w + c_0)
  $$
- Special case (plane at infinity) reduces to $d = 1/z$.  
- Provides flexibility for stereo vision and 3D reconstruction, especially in **plane plus parallax** formulations.  
- Enables algorithms to sample 3D space in a way that best matches observed image motions.

## See Also
- [[Understanding Projective Depth]]
- [[Inverse Depth]]
- [[Camera Matrix]]
- [[Object Centered Projection]]
- [[Para-Perspective Projection]]
- [[Perspective Projection]]