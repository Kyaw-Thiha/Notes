# Denoising Models

### Tier-1 — proven / SOTA-ish
| Name (with year) | Very short concise desc | paper link(s) |
|---|---|---|
| HSDT (ICCV 2023) | Hybrid CNN-Transformer; guided spectral self-attention | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Lai_Hybrid_Spectral_Denoising_Transformer_with_Guided_Attention_ICCV_2023_paper.html) · [PDF](https://openaccess.thecvf.com/content/ICCV2023/papers/Lai_Hybrid_Spectral_Denoising_Transformer_with_Guided_Attention_ICCV_2023_paper.pdf) |
| SERT (CVPR 2023) | Rectangle self-attention + spectral enhancement | [CVF PDF](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Spectral_Enhanced_Rectangle_Transformer_for_Hyperspectral_Image_Denoising_CVPR_2023_paper.pdf) · [arXiv](https://arxiv.org/abs/2304.00844) |
| SST (AAAI 2023) | Spatial-spectral transformer; non-local similarity | [AAAI page](https://ojs.aaai.org/index.php/AAAI/article/view/25221) · [PDF](https://ojs.aaai.org/index.php/AAAI/article/view/25221/24993) |
| HIDER (TNNLS 2024) | U-shaped 3D transformer; spatial–spectral constraints | [TNNLS/DOI info](https://www.researchgate.net/publication/364964736_Hider_A_Hyperspectral_Image_Denoising_Transformer_With_Spatial-Spectral_Constraints_for_Hybrid_Noise_Removal) |
| DNA-Net (TGRS 2025) | Deep-unfolding Transformer; local+non-local+spectral | [arXiv](https://arxiv.org/abs/2305.04047) · [OpenReview](https://openreview.net/forum?id=FRUi7AewaS) |

### Tier-2 — newer / promising
| Name (with year)                  | Very short concise desc                                       | paper link(s)                                                                                                                                                                                                        |
| --------------------------------- | ------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| SSRT-UNet (2024)                  | Spatial-spectral *recurrent* transformer block                | [arXiv](https://arxiv.org/abs/2401.03885)                                                                                                                                                                            |
| HSSD (2024)                       | Hybrid spatial-spectral dual-path (CNN+Transformer)           | [arXiv](https://arxiv.org/abs/2406.08782)                                                                                                                                                                            |
| HDST (2025)                       | Hybrid-domain (spatial+frequency+channel) transformer         | [arXiv](https://arxiv.org/abs/2507.20099) · [MDPI (apps)](https://www.mdpi.com/2076-3417/15/17/9735)                                                                                                                 |
| LaMamba (TGRS 2025)               | State-space (Mamba) with linear attention for HSI             | [ADS/abstract](https://ui.adsabs.harvard.edu/abs/2025ITGRS..63S3739D/abstract) · [ResearchGate](https://www.researchgate.net/publication/395812086_LaMamba_Linear_Attention_Mamba_for_Hyperspectral_Image_Denoising) |
| HSIDMamba / HSDM (2024)           | Bidirectional state-space (Mamba) for denoising               | [arXiv](https://arxiv.org/abs/2404.09697)                                                                                                                                                                            |
| VISIONARY / SSCformer (WACV 2025) | Spatial-Spectral-Cubic Transformer + Global Feature Attention | [CVF](https://openaccess.thecvf.com/content/WACV2025/html/Dixit_VISIONARY_Novel_Spatial-Spectral_Attention_Mechanism_for_Hyperspectral_Image_Denoising_WACV_2025_paper.html)                                         |
| TD-SAT (2024)                     | 3D spatial-spectral attention transformer                     | [ResearchGate](https://www.researchgate.net/publication/383968503_Three-Dimension_Spatial-Spectral_Attention_Transformer_for_Hyperspectral_Image_Denoising)                                                          |
| HTD-Mamba(2024)                   |                                                               | [Arvix](https://arxiv.org/abs/2407.06841)                                                                                                                                                                            |

### Tier-3 — compact CNN baselines (max 3)
| Name (with year) | Very short concise desc | paper link(s) |
|---|---|---|
| HSID-CNN (TGRS 2019) | Spatial-spectral residual CNN baseline | [IEEE/refs via code](https://github.com/qzhang95/HSID-CNN) |
| QRNN3D (TNNLS 2020) | 3D conv + quasi-recurrent pooling across bands | [arXiv](https://arxiv.org/abs/2003.04547) |
| Single-Model CNN (TGRS 2020) | Simple unified CNN; strong classical baseline | [PDF](https://www2.umbc.edu/rssipl/people/aplaza/Papers/Journals/2020.TGRS.Denoising.pdf) |

