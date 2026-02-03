# Deformable Convolutional Networks

`DCN` improves [[Convolutional Neural Network (CNN)|CNN]] by allowing them to better handle changes in shapes and position of objects in images.

![Deformable Convoluational Network|400](https://miro.medium.com/v2/1*oxya29Xsd1qeSjbAeyr7mw.png)

---
## Main Idea
[[Convolutional Neural Network (CNN)|CNNs]] have a fundamental limitation in capturing `geometric transformations` due to their reliance on fixed spatial structures.

The paper proposes two modules
- `Deformable Convolution`
- `Deformable ROI pooling`

which modify the spatial sampling locations by introducing `learnable offsets`.

These learnable offsets enhance the network's ability to model complex transformations without requiring additional supervision.

---
## Background Information
A critical challenge in visual recognition systems is effectively handling geometric variations.

![Convolution not learning irregular geometry|300](https://miro.medium.com/0*U9bPmjN-iAvhtuNS)

Traditionally, this has been approached by either 
- `augmenting` training datasets
- employing `transformation-invariant` features 
  such as `SIFT` or `sliding window techniques`

But both methods have limitations
- Relying on predetermined `geometric transformations`
- Difficulty in designing `robust invariant features` for complex geometries

---
## Deformable Convolution
`Deformable Convolution` introduces 2D offsets to regular grid sampling locations.

![Deformable Convolution|400](https://miro.medium.com/v2/1*6lBZ5rM1fExa_N_VTtNfXw.png)

[[Deformable Convolution|Read More]]

---
## Deformable RoI Pooling
`Deformable RoI pooling` enhances flexibility by incorporating learned offsets to dynamically adjust the bin positions of [[Pooling]].

![Deformable RoI Pooling|300](https://miro.medium.com/v2/resize:fit:1400/1*flnN8-vVlJJ--AEqDfkFjg.png)

[[Deformable RoI Pooling|Read More]]

---
## See Also
- [Deformable Convolutional Networks(2017)](https://arxiv.org/abs/1703.06211)
- [Blog explanation](https://towardsdatascience.com/deformable-convolutions-demystified-2a77498699e8/)

