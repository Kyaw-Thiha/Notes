# CFAR
#radar/cfar

`CFAR` is an adaptive-threshold detector ensuring constant false alarm rate under unknown or varying clutter.

![CFAR|400](https://i.ytimg.com/vi/BEg29UuZk6c/hq720.jpg?sqp=-oaymwEhCK4FEIIDSFryq4qpAxMIARUAAAAAGAElAADIQj0AgKJD&rs=AOn4CLCqFZxhp1-g1pprjiMyXhyk53lK_A)


---
## Background Information
In radar detection, the objective is to decide whether a given `Range-Doppler` cell contains a target or only noise/clutter.

Since clutter levels vary spatially and temporally, a `fixed detection threshold` is unusable.

---
## CFAR Mechanism
1. Select a test cell (`CUT`)
2. Gather a neighbouring cells
3. Estimate local clutter statistics
4. Compute the `adaptive threshold`
$$
T = \alpha.\hat{Z}
$$

where
- $\hat{Z}$ is the `clutter estimate`
- $\alpha$ is the parameter to control false alarm rate

5. Compare the `CUT` amplitude to threshold.
$$
|x_{CUT}|^2 > T
$$

---
## Classic Algorithms
### CA-CFAR (Cell Averaging CFAR)
Assumes homogeneous clutter and averages reference cells.
- Performs well in homogeneous noise/clutter.
- Highly sensitive to interfering targets within reference window

Used in industry due to fast speed.

### OS-CFAR (Order Statistics CFAR)
Instead of averaging, `OS-CFAR` sorts reference cells and selects a statistic $(\text{E.g: } k^{th} \text{ largest})$.
- Excellent in multi-target or heterogeneous noise.
- Longer processing time than `CA-CFAR`.

Used in academic due to high quality.

### GO-CFAR (Greatest of CFAR)
Designed for non-homogeneous clutter $(\text{such as edges or transitions})$.
`GO-CFAR` handles clutter edges by taking the maximum of sub-window estimates.

### SO-CFAR (Smallest of CFAR)
Designed for non-homogeneous clutter $(\text{such as edges or transitions})$.
`SO-CFAR` is useful when strong interference exists on one side.

---
## Advanced Algorithms
- Advanced `CFAR`  better handles non-stationary clutter.
- Modern research expands `CFAR` to heavy-tailed, multimodal clutter distributions and imaging radars

---
## See Also
- [Youtube Video](https://youtu.be/BEg29UuZk6c?si=r80fki8NcniYm1Z_)
