# Denoising
#cv/tasks/denoising
`Denoising` remove noise from the image.

![Image Denoising](https://upload.wikimedia.org/wikipedia/commons/e/e8/ROF_Denoising_Example.png)

---
## Problem
Images acquired in real-world conditions are corrupted by noise.
These are random variations in pixel values arising from
- photon shot noise, 
- sensor thermal noise, 
- quantization artifacts, or 
- compression.

$$
y = x + \epsilon 
\quad , \ \epsilon \sim \mathcal{N}(0, \sigma^2I)
$$

The goal of denoising is to recover a clean image $\mathbf{x}$ form a noisy observation $\mathbf{y}$.

---
