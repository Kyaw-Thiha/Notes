# CVAE Variants


| Name                                                               | Very short desc                       | Paper link(s)                             |                                  |
| ------------------------------------------------------------------ | ------------------------------------- | ----------------------------------------- | -------------------------------- |
| CVAE (Sohn et al., 2015)                                           | Condition encoder+decoder on c        | https://arxiv.org/abs/1506.02158          |                                  |
| β-VAE (Higgins et al., 2017)                                       | KL↑ to encourage disentanglement      | https://openreview.net/forum?id=Sy2fzU9gl |                                  |
| Capacity schedule (Burgess et al., 2018)                           | Gradually raise KL “capacity”         | https://arxiv.org/abs/1804.03599          |                                  |
| β-TCVAE (Chen et al., 2018)                                        | Penalize total correlation            | https://arxiv.org/abs/1802.04942          |                                  |
| FactorVAE (Kim & Mnih, 2018)                                       | Adversarial TC penalty                | https://arxiv.org/abs/1802.05983          |                                  |
| DIP-VAE (Kumar et al., 2018)                                       | Match moments to factorized prior     | https://openreview.net/forum?id=H1kG7GZAW |                                  |
| InfoVAE / MMD-VAE (Zhao et al., 2017)                              | Keep MI; MMD to avoid collapse        | https://arxiv.org/abs/1706.02262          |                                  |
| VampPrior (Tomczak & Welling, 2018)                                | Learned pseudo-input mixture prior    | https://arxiv.org/abs/1705.07120          |                                  |
| IAF / flow posteriors (Kingma et al., 2016)                        | Flow-based flexible q(z)              | https://arxiv.org/abs/1606.04934          | https://arxiv.org/abs/1606.04934 |
| CF-VAE (Bhattacharyya et al., 2019)                                | Conditional flow prior for CVAE       | https://arxiv.org/abs/1912.07549          |                                  |
| Ladder VAE (Sønderby et al., 2016)                                 | Hierarchical stochastic layers        | https://arxiv.org/abs/1602.02282          |                                  |
| NVAE (Vahdat & Kautz, 2020)                                        | Deep hierarchical VAE (high-fidelity) | https://arxiv.org/abs/2007.03898          |                                  |
| LDVAE for HSI unmixing (Mantripragada & Qureshi, 2022; +2023 ext.) | Dirichlet bottleneck for abundances   | https://arxiv.org/abs/2204.10869          |                                  |
| Self-Attention CVAE for HSI (Chen et al., 2021)                    | Spectral attention + CVAE             | https://www.mdpi.com/2072-4292/13/6/1072  |                                  |

**A few modern variants to also skim (≤3):**

| Name | Very short desc | Paper link(s) |
|---|---|---|
| iVAE (Khemakhem et al., 2020) | Identifiable latents via auxiliary vars | https://arxiv.org/abs/2006.00813 |
| VDVAE (Child, 2021) | Very deep hierarchical VAE | https://arxiv.org/abs/2011.10650 |
| BIVA (Maaløe et al., 2019) | Bidirectional inference; many layers | https://arxiv.org/abs/1902.02102 |

