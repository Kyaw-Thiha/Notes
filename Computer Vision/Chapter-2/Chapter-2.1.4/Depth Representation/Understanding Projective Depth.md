# Understanding Projective Depth with Matrix Examples

The formula for projective depth is  

$$
d = \frac{s_3}{z} \, (\hat{n}_0 \cdot p_w + c_0)
$$  

where:  
- $p_w$: 3D point in world coordinates.  
- $c$: camera center.  
- $r_z$: optical axis (third row of rotation matrix $R$).  
- $z = r_z \cdot (p_w - c)$: depth along the camera’s optical axis.  
- $\hat{n}_0, c_0$: reference plane parameters.  

This means: **projective depth $d$ measures how far the point lies from a chosen plane, normalized by its distance from the camera.**

---

## 1. Setup Example

Suppose:  

- Camera center at the origin:
  $$
  c = 
  \begin{bmatrix}
  0 \\ 0 \\ 0
  \end{bmatrix}
  $$
- Camera aligned with world axes:
  $$
  R = I =
  \begin{bmatrix}
  1 & 0 & 0 \\
  0 & 1 & 0 \\
  0 & 0 & 1
  \end{bmatrix}
  $$
  so $r_z = (0,0,1)^T$.
- 3D point in the world:
  $$
  p_w =
  \begin{bmatrix}
  2 \\ 3 \\ 4
  \end{bmatrix}
  $$
- Reference plane:
  $$
  \hat{n}_0 = 
  \begin{bmatrix}
  0 \\ 0 \\ 1
  \end{bmatrix}, \quad
  c_0 = -2
  $$
  This corresponds to the plane $z - 2 = 0$, i.e. $z=2$.

- Scale: $s_3 = 1$.

---

## 2. Step-by-Step Computation

1. **Compute $z$ (depth along optical axis):**

   $$
   z = r_z \cdot (p_w - c)
   =
   \begin{bmatrix}
   0 & 0 & 1
   \end{bmatrix}
   \begin{bmatrix}
   2 \\ 3 \\ 4
   \end{bmatrix}
   = 4
   $$

   So the point lies **4 units in front of the camera**.

---

2. **Compute distance from reference plane:**

   $$
   \hat{n}_0 \cdot p_w + c_0
   =
   \begin{bmatrix}
   0 & 0 & 1
   \end{bmatrix}
   \begin{bmatrix}
   2 \\ 3 \\ 4
   \end{bmatrix}
   + (-2)
   = 4 - 2 = 2
   $$

   So the point is **2 units above the reference plane $z=2$**.

---

3. **Compute projective depth $d$:**

   $$
   d = \frac{s_3}{z} (\hat{n}_0 \cdot p_w + c_0)
   = \frac{1}{4} (2) = 0.5
   $$

   Final result: $d = 0.5$.

---

## 3. Interpretation

- If we had chosen the **reference plane at infinity** ($\hat{n}_0 = 0, c_0 = 1$), then

  $$
  d = \frac{1}{z}
  $$
  which would give $d = \tfrac{1}{4} = 0.25$.  
  → This reduces to **inverse depth**.

- With a finite reference plane, projective depth tells us how far the point lies relative to that plane, normalized by its camera depth.  
  In this example, the point at $(2,3,4)$ lies *halfway above the chosen plane $z=2$, compared to its optical depth $z=4$*.

---
## 4. Summary

- $z$: depth along optical axis.  
- $\hat{n}_0 \cdot p_w + c_0$: signed distance from the reference plane.  
- Projective depth $d$ = **(distance from reference plane) ÷ (camera depth)**.  
- Special case: plane at infinity $\Rightarrow d=1/z$ (inverse depth).

## See Also
- [[Projective Depth]]