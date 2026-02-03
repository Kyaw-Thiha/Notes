# RODNet
#### A Real-Time Radar Object Detection Network Cross-Supervised by Camera-Radar Fused Object 3D Localization

`RODNet` is a [[Radar|radar]] object detection network which is cross-supervised by `Camera-Radar-Fused algorithm` during training.

![image|400](https://notes-media.kthiha.com/RODNet-(2021)/c64cd28871cbc93d8febd90959a4d50e.png)

- Its architecture is made of [[Convolutional Neural Network (CNN)|3D CNN]] with `hourglass(HG) architecture` with skip connections.
- A `chirp merging module` (`M-Net`) is proposed to combine the chirp-level features into the frame-level features.
- A `temporal deformable convolution` (`TDC`) is proposed to handle the change in radar reflection pattern over time.

---
## Radar Signal Processing
The radar data is represented in `Range-Azimuth-Doppler` format.

![image|500](https://notes-media.kthiha.com/RODNet-(2021)/a855adc5c3d46505ac5f1d593b90f520.png)

- [[FMCW|FMCW radar]] transmits continuous chirps.
- Then, it receive reflected echoes from the obstacles.
- After pre-processing, [[Fast Fourier Transform (FFT)|FFT]] is applied on samples to estimate `range` of reflections.
- A `low pass filter (LPF)` is used to remove high-frequency noise across all chirps in each frame.
- A second [[Fast Fourier Transform (FFT)|FFT]] is applied to estimate the `azimuth angle` of the reflections.

---
## Features of Radar Signal
The radar signals have following properties:
- `Rich motion information`
	- The motion is collected using the `Doppler effect`.
	- The objects’ speed and variation over time is dependent on their surface texture information. 
	- Motion of `non-rigid body` $(\text{pedestrian})$ has high variance.
	- Motion of `rigid body` $(\text{car})$ is more consistent.
- `Inconsistent Resolution`
	- High resolution in `range`
	- Low resolution in `azimuth`
- `Complex Numbers`
  Frequency & phase information are represented as complex numbers.

---
## Architecture

![RODNet|600](https://notes-media.kthiha.com/RODNet-(2021)/f017c9ed60d9d346c014c112e0826fd1.png)
- (a) The `Vanilla RODNet` is a [[Convolutional Neural Network (CNN)|3D CNN]] [[AutoEncoders|autocoder]] architecture.
- (b) `Skip Connections` are added to transmit features from bottom to top layers.
- (c) `Temporal Inception Convolution Layers` are added to extract different lengths of temporal features.

`Input`
The input of the network is
$$
(C_{\text{RF}}, T, n, H, W)
$$
where
- $C_{\text{RF}}=2$ is the `no. of channels` in each complex numbered RF images
- $T$ is the `no. of RF images` in the frame
- $n$ is the `no. of chirps` in each frame
- $H$ is the `height` of RF image
- $W$ is the `width` of RF image

`Output`
The output of the network is `ConfMaps` $\hat{D}$ of shape
$$
(C_{\text{cls}}, T, H, W)
$$
where
- $C_{\text{cls}}$ is the no. of `object classes`
- $T$ is the no. of `RF images` in the frame
- $H$ is the `height` of RF image
- $W$ is the `width` of RF image

Note that `RODNet` predicts separate `ConfMaps` for each object class.

---
### Objective Function
The objective function is defined as
$$
\mathcal{l} 
= \sum_{cls} \sum_{i,j} 
D^{cls}_{i,j} \ \hat{D}^{cls}_{i,j}
+ (1 - D^{cls}_{i,j}) \ \log(1 - \hat{D}^{cls}_{i,j})
$$
where
- $D$ is the `ConfMaps` generated from CRF annotations
- $\hat{D}$ is the the `predicted ConfMaps`
- $(i,j)$ is the `pixel indices`
- $cls$ is the `class label`

---
### M-Net Module
The `M-Net` combine chirp level features into frame-level features.

- `Input` is the RF images of 1 frame with $n$ chirps $(C_{RF}, n, H, W)$
- To merge features $n$ chirps into $1$, a [[Pooling|temporal max pooling layer]] is applied.
- `Output` is the radar frame features of $(C_{1}, H, W)$ where $C_{1}$ is no. of `filters` for temporal convolution.

Essentially, `M-Net` acts as `Doppler compensated FFT` that can be trained end-to-end in the [[Neural Network]].

---
### Temporal Deformable Convolution
Due to the relative motion of objects, the positions of these objects within the radar's `range-azimuth` coordinates may shift over time.

This leads to variations in the reflection patterns captured in the RF images.

To tackle this, `RODNET` uses the `3D version` of [[Deformable Convolutional Networks (DCN)]].

---
## Post-Processing by Location-based NMS

### OLS

[[Non-Maximum Suppression (NMS)|Traditional NMS]] uses [[Intersection over Union (IoU)|IoU]] to filter overlapping bounding boxes.
Since `RF data` does not have bounding boxes, a new metric called `Object Location Similarity (OLS)` is created.

$$
OLS = \exp \left\{  \frac{-d^2}{2(s \ \mathcal{K}_{cls})}  \right\}
$$
where
- $d$ is the `distance between the two points` in an RF image
- $s$ is the object `distance from the radar sensor`
  It represents object scale information
- $\mathcal{K}_{cls}$ is per-class constant that represents the `error tolerance for` class $cls$

### Location-based NMS
1) Get all the 8-neighbor peaks in all $C_{cls}$ channels in $\text{ConfMaps}$ within the $3 \times 3$ window as a peak set $P=\{ p_{n} \}^N_{n=1}$
2) Pick the peak $p^* \in P$ with the highest confidence 
   Put it to the final peak set $P^*$ and remove it from set $P$.
   Calculate [[#OLS]] with each of the rest peaks $p_{i}$ $(p_{i} \neq p^*)$
3) If [[#OLS]] between $p^*$ and $p_{i}$ is greater than a threshold, remove $p_{i}$ from the peak set.
4) Repeat Steps 2 and 3 until the peak set becomes empty.

---
## See Also
- [RODNET(2021)](https://arxiv.org/abs/2102.05150)
