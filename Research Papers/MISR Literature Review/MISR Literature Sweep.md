# MISR Literature Sweep

## HSRMamba (2025)
Paper: https://arxiv.org/abs/2501.18500v2
It uses `Mamba` architecture. It have handcrafted features to address `Mamba`'s initial bad ability to understand HSI by using specific techniques for each of local & global features.
`Mamba` has capabilities to learn well on long sequences, so it might help with our large hsi images. But since we are patching hsi images when training, I am not sure how much it can help.

## ISPDiff (2024)
Paper: https://ieeexplore.ieee.org/document/10543122
This paper uses `Diffusion` architecture whereby one part of the model add noise, which the other try to remove it and get original image.
This paper propose a specific architeture of `denoising diffusion probabilistic model (DDPM)` that is capable of performing super-resolution on HSI specifically.
Note that `DDPM` is interpretable due to its probabilistic nature.

## DMSA NET (2024)
Paper: https://www.researchgate.net/publication/379715726_A_Diffusion_Model-assisted_Multi-scale_Spectral_Attention_Network_for_Hyperspectral_Image_Super-resolution
This paper also use `Diffusion`. Compared to the previous one, this one focus on the mechanism of double-step super-resolution. Also, this is pure `Diffusion`, and not `DDPM`.

## UMSFT (2024)
Paper: https://www.sciencedirect.com/science/article/abs/pii/S0030399224004900
This paper propose an unsupervised Transformer model (similar to US3RN) but with a unique approach to its queries & keys, as well as its blocks.

## SSAformer (2024)
Paper: https://www.mdpi.com/2072-4292/16/10/1766
This paper focus on fusing `Transformer` model with `CNN`, and trying to get the best performances out of `CNN Layers`.

## SSTHyper (2024)
Paper: https://openaccess.thecvf.com/content/ACCV2024/papers/Xu_SSTHyper_Sparse_Spectral_Transformer_for_Hyperspectral_Image_Reconstruction_ACCV_2024_paper.pdf
A transformer focusing on RGB to HSI. Not sure how relevant for our project. Their focus is for transformer to truly learn the continuity of spectral bands, while handling sparse nature of spatial features.

## HSST (2024)
Paper: https://www.mdpi.com/2072-4292/16/22/4127
A transformer model focusing on aerial HSI from drone imagery.
Its aerial imagery nature is relevant for our purpose.

## DMGASR (2024)
Paper: https://arxiv.org/abs/2402.17285
Interesting paper that use `AutoEncoders` to convert the HSI to latent space, before using the `Diffusion` model to denoise.
It believe that the latent space helps diffusion learn better.

## Self-supervised Spectral Super-Resolution for Fast HSI (2024)
Paper: https://www.nature.com/articles/s41598-024-81031-8
Interesting paper which focus on `Self-supervised learning` to overcome lack of abundant training data.
Not sure how good it is tho.

## HSR-Diff (2023)
Paper: https://openaccess.thecvf.com/content/ICCV2023/papers/Wu_HSR-Diff_Hyperspectral_Image_Super-Resolution_via_Conditional_Diffusion_Models_ICCV_2023_paper.pdf
Use `Conditional Diffusion Model` to automatically add gaussian noise, before denoising it back.

## 3DT-Net (2023)
Paper: https://www.sciencedirect.com/science/article/abs/pii/S1566253523002233
Use `Transformer Model` fused with `3D CNN`

Pansharpening: https://cvpr.thecvf.com/virtual/2025/poster/35263
