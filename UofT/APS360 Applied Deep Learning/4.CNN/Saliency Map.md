# Saliency Map
[[Saliency Map|Saliency maps]] use [[Backpropagation|gradients]] of the output over input to highlight the areas of the images relevant to classification.

![250](https://media.geeksforgeeks.org/wp-content/cdn-uploads/20210722235025/Saliency-maps-generated-using-Grad-CAM-Grad-CAM-and-Smooth-Grad-CAM-respectively_W640.jpg)

- Feed the image to the [[Neural Network|network]].
- Compute gradients back to the input image.
- Take maximum value of absolute gradients across channels.
- Visualize.

> **Note**
> Note that outside of intuition, these maps are not practically useful and can be misleading.

---
## See Also
- [[Convolutional Neural Network (CNN)]]
- [[Convolution Layer]]
- [[Pooling Layer]]