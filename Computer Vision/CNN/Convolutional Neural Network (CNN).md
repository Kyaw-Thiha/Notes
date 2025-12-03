# Convolutional Neural Network

`CNN` are [[Neural Network|Neural Networks]] composed of [[Convolution Layer|Convolution]] and [[Pooling]]
layers.

![CNN](https://media.licdn.com/dms/image/v2/D5612AQGOui8XZUZJSA/article-cover_image-shrink_720_1280/article-cover_image-shrink_720_1280/0/1680532048475?e=2147483647&v=beta&t=5gZVHYNL2Vc2mK3iKrpK-FcpURIFdyaP4Vi38eeeZyM)

---
### Properties
`CNNs` have 3 main properties
- `Locality`
  Nearby features matter more than far away ones
- `Translation Equivariance`
  Same patterns can appear anywhere in the image
- `Parameter Sharing`
  Instead of learning separate weights per pixel, learn a filter, and slide across the image

![CNN](https://www.louisbouchard.ai/content/images/2021/04/1_QPRC1lcfYxcWWPAC2hrQgg.gif)

---
### Layers

`Convolution Layer`

The `Convolution Layer` uses filters with hyper-parameters `filter size` $F$ and `stride` $S$ to scan across the image.
The resulting output is called `feature map` or `activation map`.

![Convolution Layer|500](https://stanford.edu/~shervine/teaching/cs-230/illustrations/convolution-layer-a.png?1c517e00cb8d709baf32fc3d39ebae67)

[[Convolution Layer|Read More]]

---

`Pooling Layer`

The `Pooling Layer` downsamples the image, carrying out `spatial invariance`.

| `Max Pooling`                                                                                                                        | `Avg Pooling`                                                                                                                                |
| ------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Select maximum value of the current filter. <br>                                                                                     | Select average value of the current filter.                                                                                                  |
| ![Max Pooling\|300](https://stanford.edu/~shervine/teaching/cs-230/illustrations/max-pooling-a.png?711b14799d07f9306864695e2713ae07) | ![Average Pooling\|300](https://stanford.edu/~shervine/teaching/cs-230/illustrations/average-pooling-a.png?58f9ab6d61248c3ec8d526ef65763d2f) |
[[Convolution Layer|Read More]]

---
`Fully Connected Layer`
The `Fully Connected Layer` is just a [[Neural Network|Dense Neural Network]] that operates on flattened input.

![Dense Layer](https://miro.medium.com/v2/1*U9nOagJUzwFUFQW30b-92Q.png)
[[Neural Network|Read More]]

---
## See Also 
- [Convolutional Neural Network Cheatsheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks)
- [[Convolution Layer]]
- [[Pooling]]