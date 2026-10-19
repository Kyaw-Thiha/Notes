# RCBEVDet++ 
## Toward High-accuracy  Radar-Camera Fusion 3D Perception Network

`RCBEVDet++` is a radar-camera fusion framework featuring radar-specific RCS-aware encoding and deformable cross-attention for spatial alignment.

![image|500](https://notes-media.kthiha.com/RCBEVDet++-(2024)/e59c5ec10bd7d978e370ba0b6b362682.png)

---
## Uniqueness from Prior Research
Compared to prior research, the key differentiaters are:

- **Radar-specific encoding**
  Prior work $(\text{like PointPillars})$ repurpose LiDAR encoders for radar.
  This paper introduces radar feature extractor `RadarBEVNet` that leverages RCS as prior for object size

- **Alignment-then-Fusion pipeline**
  Prior work $(\text{CRN})$ does fusion-then-alignment, making it more robust to radar azimuth errors
  This paper uses `Cross-Attention Multi-layer Fusion (CAMF)` with deformable attention to first align misaligned radar-camera features before fusing

- **Sparse fusion capability**
  `RCBEVDet++` extends support to query-based detection methods (like `SparseBEV`, `StreamPETR`)

---
## Architecture
![image](https://notes-media.kthiha.com/RCBEVDet++-(2024)/e59c5ec10bd7d978e370ba0b6b362682.png)

### RadarBEVNet
`RadarBEVNet` transform sparse radar point clouds into dense BEV features.
It consists of 
- dual-stream radar backbone
- RCS-aware BEV encoder

#### Dual-Stream Radar Backbone
The dual-stream capture both local & global radar features simultaneously.

![image|300](https://notes-media.kthiha.com/RCBEVDet++-(2024)/b3d1cc69250e650019354c9968ae0aa4.png)

- **Point-based Block**: captures point-level information
  It is made up of 
  1. [[Neural Network|MLP]]
  2. [[Pooling Layer|MaxPool]]
  3. [[ResNet(2015)|Residual Connection]] (concatenation with MLP output)

- **Transformer-based Block**: capture long-range contextual relationships
  It is made up of
  1. `LayerNorm`
  2. [[#Distance-Modulated Self-Attention (DMSA)|DMSA]]
  3. [[ResNet(2015)|Residual Connection]] (addition with Feature)
  4. `LayerNorm`
  5. [[Neural Network|MLP]]


##### Distance-Modulated Self-Attention (DMSA)
[[Self-Attention|Standard self-attention]] struggles to converge when points are scattered across large distances $(0 - 100 \text{m})$.
**Solution**: `DMSA` modulate attention with spatial distance to guide early training.

**Steps**
Given $N$ radar points,
1. Calculate `pairwise distances` $D \in \mathbb{R}^{N \times N}$ between all points.
2. Generate `Gaussian weight map` $G$.
$$
G_{i,j} = \exp\left( -\frac{D_{i,j}^2}{\sigma^2} \right)
$$
	where $\sigma$ is a `learnable parameter`.
3. Modulate the attention as

$$
\begin{align}
\text{DMSA} (Q, K, V)
&= \text{Softmax} \left( \frac{QK^T}{\sqrt{ d }}   
+ \log(G) \right) \cdot V \\[6pt]

&= \text{Softmax} \left( \frac{QK^T}{\sqrt{ d }}   
+ \frac{1}{\sigma^2} D^2 \right) \cdot V \\[6pt]

&= \text{Softmax} \left( \frac{QK^T}{\sqrt{ d }}   
+ \beta \  D^2 \right) \cdot V \\[6pt]
\end{align}
$$

- $\beta$ starts large (strong distance bias). 
  It decreases towards $\beta=0$ and become vanilla attention.
- Each head $i$ has its own $\beta_{i}$.
  Hence, different heads can have different receptive fields.

#### Injection & Extraction Module
Since the two backbones process feature independantly, we need to fuse them. 

![image|400](https://notes-media.kthiha.com/RCBEVDet++-(2024)/49f913eda1920c7e11dd3498550680d3.png)

At block $i$, let
- $f^i_{p}$ be the `point-based local features`
- $f^i_{t}$ be the `transformer-based global features`
##### Injection Module
The `injection module` is defined as:
$$
f^i_{p}
= f^i_{p} + \gamma \times \text{CrossAttention}( \ 
\text{LNorm}(f^i_{p}), \text{LNorm}(f^i_{t}) \ )
$$
where
- $\gamma$ is a `learnable parameter`
- $f_{p}^i$ contributes the `query`

##### Extraction Module
The `extraction module` is defined as:
$$
f_{t}^i
= \text{FFN}( \ f_{t}^i 
+ \text{CrossAttention}( \ 
\text{LNorm}(f_{t}^i), \text{LNorm}(f_{p}^i) \ )  )
$$
where
- $f_{t}^i$ contributes the `query`

#### RCS-Aware BEV Encoder
Tradition BEV encoders maps one point feature to one pixel.
`Radar Cross Section (RCS)` measures radar reflectivity with $\text{Large Object} \implies \text{High RCS}$. 

![image|400](https://notes-media.kthiha.com/RCBEVDet++-(2024)/cffa6c0a0915b2a68e3c8569b30f784a.png)

**Innovation**: Use RCS to scatter one point feature to multiple BEV pixels.

**Steps**
1. Define the `scattering radius`

$$
\text{radius}
= \sqrt{ c^2_{x} + c^2_{y} } \times V_{RCS}
$$
where the `RCS Value` $V_{RCS}$ defines the magnitude of scattering.

2. `Scatter` feature to nearby pixels.
   If multiple points scatter to same pixel $\to$ **summation pooling**
   This forms the dense BEV feature map $f_{RCS}\in R^{H\times W \times C}$ 
3. Create `Gaussian-like` weight map $G_{RCS}$
$$
G_{x,y}
= \exp\left( - \frac{(c_{x} - x)^2 + (c_{y} - y)^2}
{\frac{1}{3} (c_{x}^2 + c_{y}^2) \times V_{RCS}} \right)
$$
4. Combine the features by
$$
f'_{RCS} 
= \text{MLP}( \ \text{Concat}( \ f_{RCS}, G_{RCS} \ ))
$$
5. Feed this to an `BEV Encoder`
$$
\text{Final BEV Feature} 
= \text{BEV Encoder}(Concat(f'_{RCS}, \  \text{original BEV Feature}))
$$

### Cross-Attention Multi-layer Fusion Module


---