# Normalized Device Coordinates (NDC) and the Factor of 2
 #cv/camera/focal-length

Normalized Device Coordinates (NDC) rescale pixel coordinates into **unitless ranges**. 
The goal is to map image coordinates into a symmetric interval $[-1, 1)$, independent of resolution.

---

## 1. Basic Normalization

If we simply divide by width and height:

$$
x' = \frac{x_s}{W}, \quad y' = \frac{y_s}{H},
$$

then:

- $x' \in [0,1)$  
- $y' \in [0,1)$  

This normalizes coordinates, but the range is **$[0,1)$**, not symmetric around zero.

---

## 2. Why the Factor of 2?

To center the image coordinates around zero, we want:

- Left edge $\;\;x_s = 0 \;\;\to -1$  
- Right edge $\;x_s = W \;\;\to +1$  
- Image center $\;x_s = W/2 \;\;\to 0$

This requires scaling by **$2/W$** instead of $1/W$, leading to:

$$
x' = \frac{2x_s - W}{W}
$$

Now the range is symmetric: $x' \in [-1,1]$.

---

## 3. Aspect Ratio and General Form

In practice, we preserve aspect ratio by normalizing with the **longer dimension** $S = \max(W,H)$:

$$
x'_s = \frac{2x_s - W}{S}, \quad 
y'_s = \frac{2y_s - H}{S}
$$

- The **longer dimension** spans exactly $[-1,1]$.  
- The shorter dimension scales proportionally.  
- Aspect ratio $a = W/H$ remains intact.

---

## 4. Summary

- The **factor of 2** ensures the normalized coordinates cover $[-1,1]$, not $[0,1]$.  
- This makes the image **centered at $(0,0)$** and symmetric.  
- NDC is widely used in:
  - **Computer graphics** (e.g., OpenGL pipeline).  
  - **Computer vision** (multi-resolution processing, image pyramids).  

✅ The factor of 2 is what shifts normalization from $[0,1)$ into the symmetric range $[-1,1)$.
