# Convolution Layer
#cv/cnn

Compared to [[Neural Network|fully connected neural networks]], [[Convolutional Neural Network (CNN)|CNN]] use [[Convolution Operator|convolution kernels]] to read through the image & learn from it.

![300](https://www.louisbouchard.ai/media/ghost/content/images/2021/04/1_QPRC1lcfYxcWWPAC2hrQgg.gif)

In fully connected (dense) layers, each neuron has weight for every input & bias.
$$
y = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b
$$


In [[Convolution Layer]], a kernel is a small sets of weights shared across different parts of the kernel.

---
### Example
A $k \times k$ kernel applied to an image patch can be defined as
$$
y = \sum_{m=1}^k \sum_{n=1}^k w_{m,n} \cdot x_{i+m, \, j+n} + b
$$

So the kernel weights $(w_{m,n})$ are exactly like the [[Neural Network|neuron weights]], but applied locally.

![300](https://media.geeksforgeeks.org/wp-content/uploads/20230216175224/how-to-apply-a-2d-convolution-operation-in-pytorch.gif)

---
### Output Dimension
For each dimension $o$ of an input image with
- image dimension of size $i$
- [[Convolution Operator|kernel]] of size $k$
- [[Convolution Layer|padding]] of size $p$
- [[Convolution Layer|stride]] of size $s$

The size of output dimension is computed by 
$$
o = \frac{i + 2p - k}{s} + 1
$$
If we want to take into account of the volume, it is
$$
v = o \times o \times c_{out}
$$
where $c_{out}$ is the no. of filters.

---
#### Different Length and Width
For images with different height $H_{\text{in}}$ and width $W_{\text{in}}$, 
$$
H_{out} = \frac{H_{in} + 2p - k }{s} +1
$$
$$
W_{out} = \frac{W_{in} + 2p - k }{s} +1
$$

---
### No. of Parameters
Given a [[Convolution Layer|convolution layer]] with
- input channels: $c_{in}$
- output channels: $c_{out}$
- kernel size: $k$
- bias term

The weights per [[Convolution Operator|filter]] is
$$
k \times k \times c_{in}
$$
Hence for a layer, the total learnable weights is 
$$
\mathbf{W} = (k \times k \times c_{in} + 1) \times c_{out}
$$
where $c_{out}$ is the no. of filters.

---
### FLOPS
The [[Floating Points|FLOPs]] is defined as
$$
\text{FLOPs} \approx o^{2} \times k^{2} \times c_{in} \times c_{out}
$$

---
## Hyper-Parameters
- **Kernel Size**: e.g. $3 \times 3$, $5 \times 5$
- **Stride**: How far the filter move each step
- **Padding**: How much to pad the border with zeroes

---
### Kernel Size
**Kernel size** can be used to control whether to focus on local or global features.
- Small kernel ($3 \times 3$): look at local features like edges & textures
- Large kernel ($7 \times 7$): look at global features, capture more context

---
### Stride
**Stride** $> 1$ downsample the image by a factor of $stride$.
$$
\text{Input } 32 \times 32 + \text{Stride 2} \to \text{Output } 16 \times 16
$$

---
## Filter Size
In [[Convolution Layer]], there is usually more than 1 filter passing through it. Each filter can be thought of as learning different representations of the image.

So, each filter output their own feature maps which are stacked on top of each other to form the output of the [[Convolution Layer]].

---
## See Also 
- [Convolutional Neural Network Cheatsheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks)
- [[Convolutional Neural Network (CNN)]]
- [[Convolution Layer]]
- [[Convolution Operator]]
- [[Pooling Layer]]
