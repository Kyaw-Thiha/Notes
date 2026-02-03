# Deformable Position Sensitive RoI Pooling

`Positional Sensitive(PS) RoI Pooling` generates $k^2$ score maps for each object class.

![Deformable RoI Pooling|400](https://towardsdatascience.com/wp-content/uploads/2023/08/1WwpGqxUwPIGZTa_4PSlPJA.png)

---
## Position Sensitive RoI Pooling
Through a [[Convolution Layer|Conv Layer]], all the input feature maps are firstly converted to $k^2$ score maps for each object class.
This results in $C+1$ channels for $C$ objects.

`Score maps` are denoted as $\{ x_{ij} \}$ where $(i,j)$ enumerate all bins.

[[Pooling]] is performed on these score maps.
The output value for $(i,j)^{th}$ bin is obtained by summation from one score map $x_{ij}$ corresponding to that bin.

The difference from [[Deformable RoI Pooling|RoI Pooling]] is that a general feature map $x$ is replaced by $x_{ij}$

$$
y(i,j)
= \sum_{p \ \in \ bin(i,j)}
\frac{x_{ij}(p_{0} + p)}{n_{ij}}
$$
where
- $n_{ij}$ is the `no. of pixels` in the $i,j^{th} \text{ bin}$ 
- $\left[ i \frac{w}{k} \right] \leq p_{x} < \left[ (i+1) \frac{w}{k} \right]$ is the `width spanning` the bin
- $\left[ j \frac{h}{k} \right] \leq p_{x} < \left[ (j+1) \frac{h}{k} \right]$ is the `height spanning` the bin

---
## Deformable PS RoI Pooling
![Deformable RoI Pooling|250](https://towardsdatascience.com/wp-content/uploads/2023/08/1WwpGqxUwPIGZTa_4PSlPJA.png)

1. A [[Convolution Layer|Conv Layer]] generates the full spatial resolution offset fields.
2. For each RoI per class, `PS RoI pooling` is applied on such fields to obtain normalized offsets $\Delta \hat{p}_{ij}$.
3. Using the [[Deformable RoI Pooling#Deformable RoI Pooling Formula|this equation]], normalised offsets $\Delta \hat{p}_{ij}$ is transformed into offsets $\Delta p_{ij}$.

---
## See Also
- [[Deformable Convolutional Networks (DCN)]]
- [[Deformable RoI Pooling]]
- [[Deformable Convolution]]