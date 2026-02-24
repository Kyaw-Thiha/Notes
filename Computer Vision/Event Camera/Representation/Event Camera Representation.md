# Event Camera Representation
An `event` is denoted as a tuple $(\mathbf{u}, t, p)$.
This is triggered when `logarithmic intensity` $L$ surpasses the threshold $C$.

![Event Camera|400](https://miro.medium.com/v2/1*7U8G15lbNxnqg0-H2iFP1w.jpeg)

This can be denoted as
$$
p = \begin{cases}
+1 & , \ L(\mathbf{u}, t) - L(\mathbf{u}, t - \Delta t) \geq C \\[6pt]
-1 & , \ L(\mathbf{u}, t) - L(\mathbf{u}, t- \Delta t) \leq -C \\[6pt]
0 &, \ \text{other}

\end{cases}
$$
where 
- $\mathbf{u} = (x, y)$ is the `pixel location`
- $t$ is the `timestamp`
- $p\in \{ -1,1 \}$ is the `polarity` $(\text{sign of brightness})$
- $\Delta t$ is the `time interval` since last event at pixel $\mathbf{u} = (x,y)$

**Stream of Events**
A stream of events can then be denoted as
$$
\mathcal{E}
= \{ e_{i} \}^N_{i=1}
= \{ \mathbf{u}_{i}, \ t_{i}, \ p_{i} \}
, \ i \in N
$$

---
### Different Representation Methods
**Image-Based**
Seamless integration with existing computer vision tools.
But loses fine temporal structure.
[[Event Camera Image Representation|Read More]]

**Surface-Based** (SAE)
Preserves local spatiotemporal context
But has normalization complexity
[[Event Camera Surface Representation|Read More]]

**Voxel-Based**
Enhanced temporal resolution and explicit 3D structure.
But memory and compute scales with bins.
[[Event Camera Voxel Representation|Read More]]

**Graph-Based**
Exploits sparsity and minimal compute waste
But irregular computation leads to GPU inefficiency
[[Event Camera Graph Representation|Read More]]

**Spike-Based (SNN)**
Native neuromorphic compatible.
But high training difficulty and below SOTA performance.
[[Event Camera SNN-based Method|Read More]]

**Learning-Based**
Task-adaptive and end-to-end optimized
But requires more data/compute.
[[Event Camera Learning-Based Method|Read More]]

---
## See Also
- [Paper Referenced from](https://www.semanticscholar.org/paper/Deep-Learning-for-Event-based-Vision%3A-A-Survey-and-Zheng-Liu/076fd1a9284802447eb127ff7dee8f5d4a245e69)
- [[Event Camera]]