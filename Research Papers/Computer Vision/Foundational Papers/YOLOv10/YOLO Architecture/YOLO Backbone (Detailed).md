# YOLO Backbone  
#cv/object-detection #yolo/backbone  

> The **backbone** is YOLO’s feature extractor. It transforms the input image into a **hierarchy of feature maps** at increasing receptive fields and strides. 
> These multi-scale features (e.g., $P_3$, $P_4$, $P_5$ …) are consumed by the [[YOLO Neck (Detailed)]] and predicted on by the [[YOLO Head]].

---

## What the Backbone Does
- **Early stem**: aggressively reduces spatial size while expanding channels (e.g., stride-2 conv or Focus/Space-to-Depth).
- **Stacked stages**: blocks with residual/partial connections that deepen capacity and grow receptive field.
- **Top-of-backbone context**: spatial pyramid pooling (SPP/SPPF) to mix multi-scale context before handing features to the neck.

Typical output feature strides:
- $P_3$: stride $8$ (good for small objects)  
- $P_4$: stride $16$  
- $P_5$: stride $32$  
- (Optional) $P_6$: stride $64$ for very large inputs/objects

> [!note] Stride vs. downsampling depth
> If each downsampling stage halves spatial size, the cumulative output stride after $n$ downsamples is $s = 2^n$.

---

## Canonical Building Blocks
- **Conv-BN-Activation (CBA)**: $1{\times}1$ and $3{\times}3$ kernels dominate; group/depwise variants appear in mobile settings.  
  - Common activations across YOLO family: **LeakyReLU**, **Mish** (some v4 configs), **SiLU/Swish** (v5+), **ReLU**/**ReLU6** for mobile.
- **Residual Bottleneck**: short skip to stabilize deep training and encourage gradient flow.
- **CSP (Cross-Stage Partial)**: split feature map; transform one path; **concat** with bypass to reduce compute while preserving gradients.
- **ELAN / E-ELAN**: extended layer aggregation to balance depth/width and maintain gradient paths (popularized around YOLOv7).
- **C2f** (Cross-Stage “faster”/“full” connections): a lightweight, high-gradient-flow alternative to classic CSP bottlenecks (common in YOLOv8).
- **RepConv / Rep blocks**: train-time multi-branch convs re-parameterized to a single conv at inference for speed.
- **SPP / SPPF**: multi-kernel pooling (or fused fast variant) at the top of the backbone for long-range context.

---

## A (Very) High-Level Evolution Map
| YOLO Generation | Representative Backbone Idea |
|---|---|
| **YOLOv3** | Darknet-53 (residual bottlenecks + $3{\times}3/1{\times}1$ convs) |
| **YOLOv4 / Scaled-YOLOv4** | **CSPDarknet-53** (+ SPP); extensive bag-of-freebies/tricks |
| **YOLOv5** | CSP-style backbone; **Focus**/Space-to-Depth stem (early versions), **SPPF** on top |
| **YOLOv7** | **E-ELAN** aggregation with reparam tricks; strong training heuristics |
| **YOLOv8** | CSP-style backbone with **C2f** modules; SiLU activations; SPPF |
| **P6/P7 variants** | Add deeper stages for $P_6$/$P_7$ outputs when training on larger inputs |

> [!tip] Don’t overfit to names  
> “Darknet”, “CSP”, “ELAN”, “C2f” are *design motifs*. The practical questions are: **Which strides do you emit?** **How deep/wide is each stage?** **What’s your activation and normalization?** **Do you include SPPF?**

---

## Anatomy of a YOLO-Style Backbone
**Stem (Stage 1)**  
- Either a single stride-2 $3{\times}3$ conv, or a **Focus / Space-to-Depth** step followed by conv.  
- Goal: shrink $H{\times}W$ early, expand channels, keep aliasing minimal.

**Intermediate Stages (2–4)**  
- Each stage: downsample (usually stride-2 $3{\times}3$ conv) → $k$ stacked blocks (Residual/CSP/C2f/ELAN).  
- Channel width typically doubles each downsample; block depth grows with model size (e.g., n/s/m/l/x).

**Top-of-Backbone (Stage 5)**  
- Final block stack for strong semantics.  
- **SPP/SPPF** injects multi-scale context before features exit to the [[YOLO Neck (Detailed)]].

---

## Outputs & Interface with the Neck
- Expose **multiple pyramid levels** (e.g., $P_3$, $P_4$, $P_5$; optionally $P_6$).  
- These are **tap points** taken after each major stage (pre- or post-SPPF depending on design).  
- The [[YOLO Neck (Detailed)]] (e.g., PAN/FPN/BiFPN/FPD across versions) merges them bottom-up & top-down.

```
Input
  └─ Stem (↓2)
     └─ Stage2 blocks (↓2)  → P3 (s=8)
        └─ Stage3 blocks (↓2)  → P4 (s=16)
           └─ Stage4 blocks (↓2)  → P5 (s=32)
              └─ SPP/SPPF
```

---

## Design Dials You’ll Actually Tune
- **Model scale (n/s/m/l/x)**: width & depth multipliers.  
- **Strides emitted**: train on small objects? ensure strong $P_3$ (and consider $P_2$ with stride $4$).  
- **Activation**: SiLU is a solid default; ReLU6 for mobile; LeakyReLU on older configs.  
- **Normalization**: BatchNorm is standard; consider SyncBN for multi-GPU, or GroupNorm for small batch.  
- **Downsample schedule**: Don’t over-downsample early if your data has tiny objects.  
- **SPPF on/off**: usually **on**; cheap context with measurable gains.  
- **Rep-param**: enable in training; **export** a re-parameterized model for deployment.

---

## Minimal Pseudocode (PyTorch-like)
```python
class CBA(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, bias=False)
        self.bn   = nn.BatchNorm2d(c_out)
        self.act  = nn.SiLU(inplace=True) if act else nn.Identity()
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class C2fBlock(nn.Module):
    # Simplified: split + lightweight cross-stage fusion
    def __init__(self, c, n=3):
        super().__init__()
        self.cv1 = CBA(c, c)
        self.m   = nn.Sequential(*[CBA(c, c) for _ in range(n)])
        self.cv2 = CBA(2*c, c)  # fuse
    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([y1, y2], 1))

class SPPF(nn.Module):
    def __init__(self, c, k=5):
        super().__init__()
        self.cv1 = CBA(c, c//2, 1, 1, 0)
        self.cv2 = CBA(2*(c//2), c, 1, 1, 0)
        self.k = k
    def forward(self, x):
        x = self.cv1(x)
        y1 = F.max_pool2d(x, self.k, stride=1, padding=self.k//2)
        y2 = F.max_pool2d(y1, self.k, stride=1, padding=self.k//2)
        y3 = F.max_pool2d(y2, self.k, stride=1, padding=self.k//2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))

class YOLOBackbone(nn.Module):
    def __init__(self, ch=3, widths=(64,128,256,512), blocks=(2,4,6,2)):
        super().__init__()
        w1,w2,w3,w4 = widths
        b1,b2,b3,b4 = blocks
        self.stem = CBA(ch, w1, k=3, s=2, p=1)        # ↓2
        self.s2   = nn.Sequential(CBA(w1,w2,3,2,1),   # ↓4
                                  *[C2fBlock(w2) for _ in range(b1)])
        self.s3   = nn.Sequential(CBA(w2,w3,3,2,1),   # ↓8
                                  *[C2fBlock(w3) for _ in range(b2)])
        self.s4   = nn.Sequential(CBA(w3,w4,3,2,1),   # ↓16
                                  *[C2fBlock(w4) for _ in range(b3)])
        self.sppf = SPPF(w4)                          # context (≈P5)
    def forward(self, x):
        x = self.stem(x)
        x = self.s2(x); p3 = x
        x = self.s3(x); p4 = x
        x = self.s4(x); x  = self.sppf(x); p5 = x
        return p3, p4, p5
```

---

## Practical Tips
- **Small objects**: consider exposing $P_2$ (stride $4$) or increasing $P_3$ channel width; avoid excessive early downsampling.  
- **Throughput**: use Rep* blocks and export re-parameterized weights; prefer SPPF over SPP for cheap gains.  
- **Memory**: C2f/CSP modules reduce compute vs. naive residual stacks at comparable accuracy.  
- **Transfer learning**: load a **backbone-only** checkpoint to warm-start training on a new dataset.  
- **Ablations to try**: SiLU vs. LeakyReLU; with/without SPPF; C2f depth; adding/removing $P_6$ output.

---

## See Also
- [[YOLO Family Architecture]] — how backbone, neck, and head fit together  
- [[YOLO Neck (Detailed)]] — feature aggregation (PAN/FPN/…)
- [[YOLO Head]] — decoupled classification/regression heads & loss design
