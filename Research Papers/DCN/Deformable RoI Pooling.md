# Deformable RoI Pooling
`Deformable RoI pooling` enhances flexibility by incorporating learned offsets to dynamically adjust the bin positions of [[Pooling]].

![Deformable RoI Pooling|300](https://miro.medium.com/v2/resize:fit:1400/1*flnN8-vVlJJ--AEqDfkFjg.png)

---
## RoI Pooling

Let
- $x$ be the `input feature map`
- $w \times h$ be the `size of RoI`
- $p_{0}$ be the `top-left corner`
- $k$ be the hyperparam for `pooling size`
- $(0\leq i, \ j < k)$ be the $(i,j)^{th}$ bin


Then, `RoI Pooling` divides the RoI into $k \times k$ bins, and outputs a $k \times k$ feature map $y$.

Hence for the $(i,j)^{th}$ bin,
$$
y(i,j)
= \sum_{p \ \in \ bin(i,j)}
\frac{x(p_{0} + p)}{n_{ij}}
$$
where
- $n_{ij}$ is the `no. of pixels` in the $i,j^{th} \text{ bin}$ 
- $\left[ i \frac{w}{k} \right] \leq p_{x} < \left[ (i+1) \frac{w}{k} \right]$ is the `width spanning` the bin
- $\left[ j \frac{h}{k} \right] \leq p_{x} < \left[ (j+1) \frac{h}{k} \right]$ is the `height spanning` the bin

---
### Deformable RoI Pooling Formula
To implement `deformable RoI pooling`, offsets $\{ \Delta p_{ij} \mid 0 \leq i,j < k \}$ are added:
$$
y(i,j) 
= \sum_{p \in bin(i,j)}
\frac{x(p_{0} + p + \Delta p_{ij})}{n_{ij}}
$$

---
### Algorithm
The network architecture to achieve `deformable RoI pooling` is shown below.
![Deformable RoI Pooling|300](https://www.oejournal.org/fileOEJ/journal/article/gdgc/2019/9/gdgc-46-9-180606-1-4.jpg)

1. `RoI Pooling` generates the pooled feature map.
2. From the maps, a [[Neural Network|Fully-Connected Layer]] generates the normalized offsets $\Delta \hat{p}_{ij}$
3. Using the [[#Deformable RoI Pooling Formula|equation above]], normalised offsets $\Delta \hat{p}_{ij}$ is transformed into offsets $\Delta p_{ij}$.
$$
\Delta p_{ij}
= \gamma \cdot \Delta \hat{p}_{ij} \circ (w,h)
$$
   This is done by element-wise product with the RoI’s width and height.
   
   Note that
	- $\gamma$ is a pre-defined scalar to modulate the magnitude of the offset.
	- `offset normalization` is necessary to make the offset learning invariant to RoI size

---
## Backpropagation
The [[Backpropagation]] of `deformable RoI Pooling` can be represented as

$$
\begin{aligned}
&\frac{\partial y(i,j)}{\partial \Delta p_{ij}}
\\[6pt]
&= \frac{1}{n_{ij}} \sum_{p \in \mathrm{bin}(i,j)}
\frac{\partial \ x\!\left(p_{0} + p + \Delta p_{ij}\right)}
{\partial \Delta p_{ij}}
\\[6pt]
&= \frac{1}{n_{ij}} \sum_{p \in \mathrm{bin}(i,j)}
\frac{\partial}{\partial \Delta p_{ij}}
\sum_{q} G\!\left(q,\ p_{0} + p + \Delta p_{ij}\right)\, x(q)
\\[6pt]
&= \frac{1}{n_{ij}} \sum_{p \in \mathrm{bin}(i,j)}
\left[
\sum_{q}
\frac{\partial \ G\!\left(q,\ p_{0} + p + \Delta p_{ij}\right)}
{\partial \Delta p_{ij}}
\ \cdot\ x(q)
\right].
\end{aligned}
$$

---
## See Also
- [[Deformable Convolutional Networks (DCN)]]
- [[Deformable Position Sensitive RoI Pooling]]
- [[Deformable Convolution]]