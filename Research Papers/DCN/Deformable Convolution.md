# Deformable Convolution
`Deformable Convolution` introduces 2D offsets to regular grid sampling locations.

![Deformable Convolution|400](https://miro.medium.com/v2/1*6lBZ5rM1fExa_N_VTtNfXw.png)

---
## Problem Formulation
Let grid $\mathcal{R}$ defines the `receptive field size & dilation`.
For $3 \times 3 \text{ kernel}$ with $\text{dilation}=1$,
$$
\mathcal{R}
= \{ 
(-1, -1), \ (-1, 0), \ \dots, \ (0,1), \ (1,1) 
\}
$$

`Convolutional Layer`
For each location $p_{0}$ on the `output feature map` $y$,
$$
y(p_{0})
= \sum_{p_{n} \in \mathcal{R}}
w(p_{n}) \ \cdot \ x(p_{0} + p_{n})
$$
where 
- $x(\cdot)$ is the `input feature map`
- $w(\cdot)$ is the `weight matrix`
- $p_{n}$ enumerates the locations in $\mathcal{R}$.

---
`Deformable Convolutional Layer`
In deformable convolution, the grid $\mathcal{R}$ is augmented with offsets:
$$
\{ \Delta p_{n} \mid n=1, \ \dots, \ N \}
$$
where $N = |\mathcal{R}|$.

Hence, we get
$$
y(p_{0})
= \sum_{p_{n} \in \mathcal{R}}
w(p_{n}) \ \cdot \ x(p_{0} + p_{n} + \Delta p_{n})
$$
where the sampling is irregular $p_{n} + \Delta p_{n}$.

---

`Bilinear Interpolation`
As the offset $\Delta p_{n}$ is typically fractional,
$$
x(p)
= \sum_{q} G(q, p) \cdot x(q)
$$
where
- $p=p_{0} + p_{n} + \Delta p_{n}$ is an arbitrary `fractional location`
- $q$ enumerates all `integral spatial locations` in the feature map $x$
- $G(\cdot, \cdot)$ is the `biliinear interpolation kernal`

$G(\cdot, \cdot)$ can separated into two one dimensional kernels:
$$
G(q, p)
= g(q_{x}, p_{x}) \ \cdot \ g(q_{y}, p_{y})
$$
where $g(a,b) = \max(0, \ 1-|a-b|)$.

---
## Algorithm
![Deformable Convolution|300](https://www.researchgate.net/publication/349646156/figure/fig5/AS:11431281387495561@1745122802177/Deformable-convolution-network.tif)

The offsets are obtained by applying a [[Convolution Layer|convolutional layer]] whose kernal is same `spatial resolution` & `dilation` as current `convolutional layer`.

The output offset fields have same spatial resolution with the input feature map.
The channel dimension $2N$ corresponds to $N$ 2D offsets.

---
## Backpropagation
The `deformable convolution` can be represented as
$$
\begin{align}
&\frac{\partial y \ (p_{0})}{\partial \Delta p_{n}}  \\[6pt]

&= \sum_{p_{n} \in \mathcal{R}} w(p_{n})
\ \cdot \  
\frac{\partial \ x(p_{0} + p_{n} + \Delta p_{n})} 
{ \partial \Delta p_{n}}
\\[6pt]

&= \frac{\partial}{\partial \Delta p_{n}} 
\sum_{q} G(q, p_{0} + p_{n} +\Delta p_{n} )
\ x(q)
\\[6pt]

&= \sum_{p_{n} \in \mathcal{R}} \left[  
w(p_{n}) \ \cdot \ \sum_{q} 
\frac{\partial \ G(q, p_{0} + p_{n} + \Delta p_{n})} 
{\partial \Delta p_{n}}
 \right]
\end{align}
$$
where
- $x(\cdot)$ is the `input feature map`
- $y(\cdot)$ is the `output feature map`
- $p_{0}$ is the `current output spatial location`
- $p_{n}$ is the $n^{th}$ `kernel sampling offset` from the regular convolution grid
- $w(p_{n})$ is the `convolution weight` associated with the kernel position $p_{n}$
- $\Delta p_{n}$ is the `learned kernal offset`
- $G(q,p)$ is the `interpolation kernel`
  How much the grid point $q$ contributes to the sampled value at the (fractional) point $p$



---
## See Also
- [Blog explanation](https://towardsdatascience.com/deformable-convolutions-demystified-2a77498699e8/)
- [[Deformable Convolutional Networks (DCN)]]
- [[Bilinear Interpolation]]
- [[Deformable RoI Pooling]]
- [[Deformable Position Sensitive RoI Pooling]]