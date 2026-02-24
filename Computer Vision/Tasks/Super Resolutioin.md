# Super Resolution
`Super-resolution (SR)` is the task of recovering a `high-resolution (HR)` image from one or more `low-resolution (LR)` observations.

![Super Resolution|300](https://miro.medium.com/v2/0*eXqvLRi6iRIG210s.jpeg)

---
## Problem Formulation
The degradation model can be formulated as
$$
\mathbf{y}
= D(\mathbf{x} \otimes \mathbf{k}) + \mathbf{n}
$$
where
- $x$ is the `high-resolution image (HR)`
- $y$ is the `low-resolution image (LR)`
- $D$ is the `downsampling operation`
- $\mathbf{k}$ is the `blur kernal`
- $\mathbf{n}$ is the `noise`

Note that many `HR images` can produce the same `LR image` under downsampling.
Hence, `SR` requires learning a prior over natural images to resolve this ambiguity.

---
