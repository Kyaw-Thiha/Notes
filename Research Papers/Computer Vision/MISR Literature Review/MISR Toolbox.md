# HSI MISR Toolbox — Quick Reference 

A compact map of *modules you can mix-and-match* for **multi-image super-resolution (MISR) on hyperspectral imagery (HSI)**, plus hand-picked papers to read next. Each section: what it is → why it helps for HSI MISR → starter papers (with links).

---

## 1) Deformable Alignment (feature-level registration & fusion)

**What it is.** Learnable alignment at the feature level (deformable convs/attention) that samples from neighboring frames at *sub-pixel offsets*; more robust than pure optical flow when motion is local or nonrigid.

**Why it helps HSI MISR.** You can fuse fine spatial detail from multiple takes while controlling spectral drift (e.g., share offsets from a high-SNR anchor band to neighbors).

**Read next (alignment patterns to borrow):**
- **BasicVSR++ (CVPR’22)** — flow-guided *deformable* alignment + second-order propagation.  
  https://openaccess.thecvf.com/content/CVPR2022/papers/Chan_BasicVSR_Improving_Video_Super-Resolution_With_Enhanced_Propagation_and_Alignment_CVPR_2022_paper.pdf
- **RVRT (NeurIPS’22)** — *guided deformable attention* inside a recurrent Transformer for video restoration.  
  https://proceedings.neurips.cc/paper_files/paper/2022/file/02687e7b22abc64e651be8da74ec610e-Paper-Conference.pdf
- **EDVR (CVPRW’19/NTIRE)** — classic PCD (Pyramid-Cascading-Deformable) alignment + temporal/spatial attention.  
  https://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Wang_EDVR_Video_Restoration_With_Enhanced_Deformable_Convolutional_Networks_CVPRW_2019_paper.pdf

---

## 2) State-Space / Transformer **Spectral Encoders**

**What it is.** Backbones that capture long-range *spatial–spectral* dependencies:
- **Transformers** (global receptive field) with spectral-aware attention.
- **State-Space Models (SSMs)** like **Mamba** with *linear-time* sequence mixing—great for many bands.

**Why it helps HSI MISR.** After alignment, you still need a cube encoder that respects inter-band structure without blowing up memory.

**Read next (drop-in spectral heads):**
- **ESSAformer (ICCV’23)** — “spectral-friendly” attention for HSI SR (good inductive biases).  
  https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_ESSAformer_Efficient_Transformer_for_Hyperspectral_Image_Super-resolution_ICCV_2023_paper.pdf
- **MSDformer (TGRS’23)** — multiscale *deformable* Transformer for HSI SR.  
  (publisher page) https://www.semanticscholar.org/paper/MSDformer%3A-Multiscale-Deformable-Transformer-for-Chen-Zhang/51ae033bf95d93b12a668bb6a23a5e7ad60f6f59
- **Mamba (Selective State Spaces, 2023)** — SSM backbone paper.  
  https://arxiv.org/abs/2312.00752
- **Vision Mamba (ICML’24)** — bidirectional Mamba for vision (reference implementation).  
  Paper: https://arxiv.org/abs/2401.09417 • Code: https://github.com/hustvl/Vim

*Implementation tip:* Start with **2D (spatial) + 1D (spectral) separable** stems or lightweight 3D conv, then stack Transformer/SSM blocks with **band-distance bias**.

---

## 3) Self-Supervision & **Test-Time Training (TTT)**

**What it is.** Improve without HR labels by exploiting physics & internal redundancy:
- **Zero-shot/internal learning** (fit a small net on the test image itself).
- **TTT** (adapt a pretrained model at inference) using LR↔HR cycle consistency, degradation modeling, spectral mixup.

**Why it helps HSI MISR.** HR HSI GT is scarce; degradations (PSF/SRF/noise) are sensor-specific. Self-sup/TTT lets you specialize *per scene*.

**Read next (ready-to-use recipes):**
- **Test-Time Training for HSI SR (2024)** — grouped upsampling + pseudo-labels + spectral mixup.  
  https://arxiv.org/abs/2409.08667
- **ZSSR (CVPR’18)** — zero-shot SR via internal learning (classic baseline to adapt).  
  https://openaccess.thecvf.com/content_cvpr_2018/papers/Shocher_Zero-Shot_Super-Resolution_Using_CVPR_2018_paper.pdf
- **Deep Image Prior (CVPR’18)** — untrained CNN as an implicit prior; useful as a physics-guided fallback.  
  https://openaccess.thecvf.com/content_cvpr_2018/html/Ulyanov_Deep_Image_Prior_CVPR_2018_paper.html

*Practical knob:* Freeze most weights; update only a **degradation branch** (PSF/SRF/noise) and the **last reconstruction stage** per test cube.

---

## 4) **Fast Diffusion Priors** (plug-and-play or few-step)

**What it is.** Use diffusion models as powerful priors for inverse problems—*but fast*:
- **DPS / posterior sampling**: guide a pretrained diffusion with your *forward model* (blur/downsample/noise).
- **Few/one-step distillation**: compress long samplers down to 1–4 steps (viable inside MISR loops).

**Why it helps HSI MISR.** Bolt a diffusion prior *after* alignment to hallucinate safe textures while constraining spectra (SAM/ERGAS/low-rank). Few-step variants keep runtime practical on large cubes.

**Read next (go-to starting points):**
- **Diffusion Posterior Sampling (ICLR’23)** — principled posterior sampling for noisy inverse problems.  
  https://arxiv.org/abs/2209.14687
- **Consistency Models (ICML’23)** — *one-step* generation / few-step sampling by design (great for speed).  
  Paper: https://arxiv.org/abs/2303.01469 • PDF: https://proceedings.mlr.press/v202/song23a/song23a.pdf
- **Progressive Distillation (2022)** — distill diffusion to very few steps.  
  https://arxiv.org/abs/2202.00512
- (Optional) **Distillation for guided diffusion (CVPR’23)** — practical speeding up of classifier-free guided models.  
  https://openaccess.thecvf.com/content/CVPR2023/papers/Meng_On_Distillation_of_Guided_Diffusion_Models_CVPR_2023_paper.pdf

**HSI-flavored diffusion to peek at (fusion, but reusable ideas):**
- **PLR-Diff (2024, Info. Fusion)** — unsupervised pansharpening with *low-rank spectral* constraints in diffusion.  
  https://doi.org/10.1016/j.inffus.2024.102583  *(search title if DOI changes)*
- **DM-ZS (CVPR’25)** — zero-shot pansharpening: frozen diffusion + *iterative spatial-spectral guidance*.  
  https://openaccess.thecvf.com (search “DM-ZS pansharpening”)

---

## How to assemble these in your MISR pipeline

1) **Align & fuse frames** with deformable alignment (BasicVSR++/RVRT patterns).  
2) **Encode spatial–spectral context** with a Transformer/SSM spectral head (ESSAformer/MSDformer or Vision-Mamba).  
3) **Adapt per scene** via TTT/self-sup using your *physics* (band-wise PSF/SRF/noise) and LR↔HR cycle consistency.  
4) **Refine with a fast diffusion prior**, guided by physics likelihood + SAM/ERGAS + low-rank penalties to keep spectra honest.

> **Quick checklist (drop into a task list):**
> - [ ] Implement DCN/def-attn alignment (PCD or guided deformable attention).
> - [ ] Add 2D+1D spectral encoder (try ESSAformer-style attention or Mamba blocks).
> - [ ] Wire TTT losses (LR↔HR cycle, spectral mixup, degradation consistency).
> - [ ] Plug a few-step / one-step diffusion head; add DPS-style physics guidance.
> - [ ] Track SAM / SID / ERGAS and band-wise uncertainty maps during eval.

---
