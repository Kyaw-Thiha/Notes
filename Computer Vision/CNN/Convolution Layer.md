# Convolution Layer
#cv/cnn

Compared to original neural networks, `CNN` used convolution kernals to 'read' through the image & learn from it.

![Convolution Layer](https://miro.medium.com/v2/resize:fit:1400/1*Q6yA_1B_vsdGWAAwB8Z7rA.png)

In fully connected (dense) layers, each neuron has weight for every input & bias.
$$
y = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b
$$


In Convolution layer, a kernal is a small sets of weights shared across different parts of the kernal.

For example, a 3×3 kernel applied to an image patch 

$$
y = \sum_{m=1}^3 \sum_{n=1}^3 w_{m,n} \cdot x_{i+m, \, j+n} + b
$$

So the kernel weights $(w_{m,n})$ are exactly like the neuron weights, but applied locally.

## Parameters
- **Kernel Size**: e.g. $3 \times 3$, $5 \times 5$
- **Stride**: How far the filter move each step
- **Padding**: How much to pad the border with zeroes

$$
H_{out} = \frac{H_{in} - K + 2P}{S} +1
$$
$$
W_{out} = \frac{W_{in} - K + 2P}{S} +1
$$

### Kernel Size
**Kernel size** can be used to control whether to focus on local or global features.
- Small kernel ($3 \times 3$): look at local features like edges & textures
- Large kernel ($7 \times 7$): look at global features, capture more context

### Stride
**Stride** $> 1$ downsample the image by a factor of $stride$.
$$
\text{Input } 32 \times 32 + \text{Stride 2} \to \text{Output } 16 \times 16
$$

## Filter Size
In `Convolution Layer`, there is usually more than 1 filter passing through it.
Each filter can be thought of as learning different representations of the image.

So, each filter output their own feature maps which are stacked on top of each other to form the output of the `Convolution Layer`.

