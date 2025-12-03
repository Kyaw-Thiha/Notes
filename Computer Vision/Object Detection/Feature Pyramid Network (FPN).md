# Feature Pyramid Network (FPN)
#cv/object-detection/fpn

`FPN` is an architecture that fuses low-level and high-level features to detect objects at different scales.

![FPN](https://miro.medium.com/v2/resize:fit:1400/1*0b5l_AhQaKjSvqY9afqzfg.png)

---
`Architecture`

A good way to visualize its architecture is to see it as a `U-Net Architecture`, consisting of a `downsampling path` and then `upsampling path`

First in the `downsampling path`, it downsamples the feature map using [[Convolution Layer]] of $stride \geq 2$. 

Second in the `upsampling path`, it upsamples the feature map using [[Interpolation Methods|Nearest Neighbour Interpolation]].
It also use `skip connections` from earlier layers.

---
`Predictions`
An important thing to note here is that `FPN` make predictions at each layer of the `upsampling path`.

The predictions in `low-resolution feature map` (earlier) represents predictions of large scale objects.
The predictions in `high-resolution feature map` (later) represents predictions of small scale objects.

---
- [[Computer Vision/Object Detection/Anchors|Anchors]]
- [[Interpolation Methods]]
