# Computing Depth Along the Camera Axis
#cv/camera/inverse-depth 

The formula  
$$
z = r_z \cdot (p_w - c)
$$  
says: *the depth $z$ of a 3D point $p_w$ is the dot product between the camera’s forward axis $r_z$ and the vector from the camera center $c$ to the point $p_w$.*

---

## 1. Components

- $p_w = (x_w, y_w, z_w)^T$: 3D point in world coordinates.  
- $c = (c_x, c_y, c_z)^T$: camera center in world coordinates.  
- $r_z$: the third row of the camera’s rotation matrix $R$, i.e., the camera’s optical axis.

If the rotation matrix is  
$$
R =
\begin{bmatrix}
r_x^T \\
r_y^T \\
r_z^T
\end{bmatrix},
$$  
then $r_z = (r_{zx}, r_{zy}, r_{zz})^T$.

---

## 2. Example in Matrix Form

Suppose:

- Camera center:  
  $$
  c =
  \begin{bmatrix}
  1 \\ 2 \\ 1
  \end{bmatrix}
  $$
- 3D point in world:  
  $$
  p_w =
  \begin{bmatrix}
  4 \\ 6 \\ 5
  \end{bmatrix}
  $$
- Camera rotation (looking along $z$ axis, no tilt/roll):  
  $$
  R =
  \begin{bmatrix}
  1 & 0 & 0 \\
  0 & 1 & 0 \\
  0 & 0 & 1
  \end{bmatrix}
  $$
  so $r_z = (0, 0, 1)^T$.

---

## 3. Step-by-Step Calculation

1. Compute the vector from camera to point:
   $$
   p_w - c =
   \begin{bmatrix}
   4-1 \\ 6-2 \\ 5-1
   \end{bmatrix}
   =
   \begin{bmatrix}
   3 \\ 4 \\ 4
   \end{bmatrix}
   $$

2. Dot with the camera’s $z$-axis:
   $$
   z = r_z \cdot (p_w - c)
   =
   \begin{bmatrix}
   0 & 0 & 1
   \end{bmatrix}
   \begin{bmatrix}
   3 \\ 4 \\ 4
   \end{bmatrix}
   = 4
   $$

So the depth of the point relative to the camera is **$z=4$**.

---

## 4. Interpretation

- If the camera is at $(1,2,1)$, looking along the global $z$ axis, then the point $(4,6,5)$ lies **4 units in front of the camera along its optical axis**.  
- If the camera were rotated, $r_z$ would be different, and the dot product would measure depth in the **rotated direction**.

---
