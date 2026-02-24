# Optical Flow Estimation
#cv/tasks/optical-flow-estimation
`Optical Flow` is the motion of pixels between the frames caused by relative motion between the camera and the scene.

![Optical Flow Estimation](https://miro.medium.com/v2/1*6px8sfSUcQvAtaIixbNYbQ.gif)

---
## Classical Problem Formulation
The change in pixels is defined as
$$
I(x, y, t)
= I(x + u, \ y + v, \ t+1)
$$

Taylor-expanding the right side, we get
$$
I_{x}u + I_{y}v + I_{t} = 0
$$
This is the `optical flow constraint equation` with two unknowns.

> Note that it assumes `brightness consistency`.
> These assumptions are replaced with `learned priors` in [[Neural Network|deep learning methods]].

---
## Applications

**Supervising training [[Monocular Depth Estimation]]**
Let 
- $D(x)$ be the known `depth`
- $(R,t)$ be the known `camera pose` between frames

Then, the optical flow can be synthesized as
$$
x' = \pi(\mathbf{R} \cdot \pi^{-1}(\mathbf{x}, D(x)), \ t)
$$
and
$$
\mathbf{w}_{geo}(x)
= \mathbf{x}' - \mathbf{x}
$$

> Note that **photo-metric warp loss** in self-supervised flow above its also the supervision signal for depth.

Specific methods use an `optical flow network` and a `depth network` together, with the constraint that their outputs must agree:
$$
\mathcal{L}_{consistency}
 = || \mathbf{w}_{flow} - \mathbf{w}_{geo} ||
$$

---
