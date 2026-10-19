## MISR (HSI)
These are models & techniques that use multiple image to perform super-resolution, and are thus baseline models that we have to aim to surpass.

Based on my research, the main barrier to MISR is the lack of data (since it is hard to burst image HSI), and not good enough ML model.
However, the latest model architectures could make this a possibility.
Namely, we should look into the following architectures
- deformable alignment
- state-space/transformer spectral encoder (like Mamba)
- self-supervision (TTT)
- fast diffusion priors

[[MISR Toolbox]]

### CNN-based MISR (2022)
Paper: https://www.researchgate.net/publication/363676089_Multiple_Frame_Splicing_and_Degradation_Learning_for_Hyperspectral_Imagery_Super-Resolution
This paper is the closest we have to `MISR` that we are doing.
They used 3D Convolution + 2D Convolution for their architecture. The main point of their paper is the `frame slicing technology`.

### Non-Deep Learning (2023)
Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC9963731/
This is a non deep-learning method that focus on low computation method, focusing on microscopic HSI imagery.

### Another non-ML based Method (2023)
Paper: https://www.sciencedirect.com/science/article/abs/pii/S0098300423001607


