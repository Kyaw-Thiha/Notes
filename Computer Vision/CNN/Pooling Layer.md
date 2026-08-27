# Pooling Layer
#cv/cnn/pooling

[[Pooling Layer|Pooling layers]] shrink the spatial size of the feature map without adding new weights.

![Pooling Layer|300](https://media.springernature.com/m685/springer-static/image/art%3A10.1038%2Fs41598-024-51258-6/MediaObjects/41598_2024_51258_Fig1_HTML.png)

The output dimension is 
$$
o = \frac{i - k}{s} + 1
$$
where
- $i$ is the image dimension
- $k$ is the [[Convolution Layer|kernel size]]
- $s$ is the [[Convolution Layer|stride size]]

---
## Different Pooling Strategies
Different pooling strategies can be used to downsample the image.

![[Pooling Strategies.png|300]]

---
## Why Pooling is useful
- **Downsampling**: Used early in the [[Convolutional Neural Network (CNN)|CNN]] to reduce computation.
- **Translation Invariance**: less sensitive to small shifts (if edge move by 1 pixel, it is still max)
- **Highlight Key Features**: Keep strongest activations (which are usually edges, corners & textures)

---
## See Also 
- [Convolutional Neural Network Cheatsheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks)
- [[Convolutional Neural Network (CNN)]]
- [[Convolution Layer]]
- [[Convolution Operator]]
- [[Pooling Layer]]