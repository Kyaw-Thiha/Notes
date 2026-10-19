# YOLO Neck  
#cv/object-detection #yolo/neck

> The **neck** fuses multi-scale features from the [[YOLO Backbone (Detailed)]] into richer, aligned pyramids that the [[YOLO Head]] can decode. 

> It performs **top-down** and often **bottom-up** aggregation with lateral connections, up/down-sampling, and conv blocks.

---

## What the Neck Does (in one screen)
- **Inputs**: a set of backbone features at strides $s \in \{8, 16, 32, (64, ...)\}$, often referred to as $P_3, P_4, P_5, (P_6, ...)$.
- **Fuse**: propagate strong semantics from high levels down, and fine detail from low levels up.
- **Outputs**: a refined pyramid (commonly 3 scales) for the head, e.g., $N_3, N_4, N_5$ at the same strides as inputs.

```
Backbone            Neck (FPN top-down + PAN bottom-up)
  P5 ───────────┐            ┌─────────── N5 (s=32)
                └─↓2 Conv────┤
  P4 ────▶ (+) ──▲─ Upsample ┴─ Conv ────┤
            │    │                        └─↓2 Conv──▶ (+) ── Conv ── N4 (s=16)
  P3 ────▶ (+) ──▲─ Upsample ── Conv ────────────────▲
            │                                       │
            └───────────────────────────────────────┴─────────────── N3 (s=8)
Legend: (+) lateral fusion (concat/sum); Upsample: ×2; ↓2 Conv: stride-2 3×3
```

---

## Canonical Neck Designs
- **FPN (Feature Pyramid Network)**  
  Top-down pathway: upsample high-level features and **lateral-merge** with same-stride lower features (via $1{\times}1$ conv alignment). Good semantics at all scales.
- **PAN / PANet (Path Aggregation Network)**  
  Adds **bottom-up** path: re-propagates fine detail upward via stride-2 fusions. Popular in YOLOv5/YOLOX-style PAFPNs.
- **PAFPN (PAN + FPN)**  
  The practical YOLO default: FPN **then** PAN with concatenation and $3{\times}3$ convs after each fusion.
- **BiFPN (Bi-directional FPN)**  
  Bi-directional multi-stage fusion with **learnable fusion weights**, often summation-based and normalized; efficient for deeper pyramids.
- **Others / Variants**  
  - **ELAN/E-ELAN aggregations** (seen with YOLOv7 motifs).  
  - **Lightweight/mobile** necks: depthwise separable or re-parameterized convs.  
  - **Extra levels**: $P_2$ (stride 4) for tiny objects; $P_6$/$P_7$ for large inputs.

---

## Core Operations & Choices
- **Lateral connections**: $1{\times}1$ conv to align channels ($C$) before fusion.
- **Fusion type**: **concat** (channel-wise) followed by $3{\times}3$ conv, or **sum** (requires channel match).  
- **Upsampling**: nearest/bilinear (common), learned upsample (e.g., sub-pixel), or CARAFE-like kernels.  
- **Downsampling**: stride-2 $3{\times}3$ conv (preferred) vs. pooling.  
- **Post-fusion blocks**: $3{\times}3$ CBA (Conv–BN–Act) stacks; RepConv in train-time, fused at export.  
- **Normalization/activation**: BatchNorm + SiLU are strong defaults.

> [!note] Channel planning
> Keep a consistent **pyramid width** (e.g., $C=256$) across scales to simplify lateral ops and the head. Mobile settings may use depthwise separable convs.

---

## Typical Shapes (for 640×640 input)
| Level | Stride | Spatial | Channels (example) |
|---|---|---:|---:|
| $P_3 \to N_3$ | 8 | $80 \times 80$ | $C$ |
| $P_4 \to N_4$ | 16 | $40 \times 40$ | $C$ |
| $P_5 \to N_5$ | 32 | $20 \times 20$ | $C$ |
| (optional $P_2 \to N_2$) | 4 | $160 \times 160$ | $C$ |

---

## Minimal Pseudocode (PyTorch-like PAFPN)
```python
class CBA(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1, act=True, dw=False):
        super().__init__()
        groups = c_in if dw else 1
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, bias=False, groups=groups)
        self.bn   = nn.BatchNorm2d(c_out)
        self.act  = nn.SiLU(inplace=True) if act else nn.Identity()
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class FPNPAN(nn.Module):
    def __init__(self, c3, c4, c5, c=256, post=1):
        super().__init__()
        # Lateral 1x1 to unify channels
        self.l5 = CBA(c5, c, k=1, p=0); self.l4 = CBA(c4, c, k=1, p=0); self.l3 = CBA(c3, c, k=1, p=0)
        # Top-down (FPN)
        self.f4 = nn.Sequential(*[CBA(2*c, c) for _ in range(post)])
        self.f3 = nn.Sequential(*[CBA(2*c, c) for _ in range(post)])
        # Bottom-up (PAN)
        self.p4 = nn.Sequential(*[CBA(2*c, c) for _ in range(post)])
        self.p5 = nn.Sequential(*[CBA(2*c, c) for _ in range(post)])
    def up2(self, x): return F.interpolate(x, scale_factor=2, mode='nearest')
    def down2(self, x): return F.max_pool2d(x, 2)  # or stride-2 conv
    def forward(self, p3, p4, p5):
        # Lateral
        n5 = self.l5(p5)
        n4 = self.l4(p4)
        n3 = self.l3(p3)
        # FPN top-down
        n4 = self.f4(torch.cat([n4, self.up2(n5)], 1))
        n3 = self.f3(torch.cat([n3, self.up2(n4)], 1))
        # PAN bottom-up
        n4 = self.p4(torch.cat([n4, self.down2(n3)], 1))
        n5 = self.p5(torch.cat([n5, self.down2(n4)], 1))
        return n3, n4, n5  # hand to YOLO Head
```

---

## Design Dials You’ll Actually Tune
- **#Levels**: $(N_3,N_4,N_5)$ vs. $(N_2\text{..}N_5)$ or $(N_3\text{..}N_6)$.  
- **Width $C$**: 128–320 typical; increase for higher-capacity heads.  
- **Fusion op**: concat+conv (more flexible) vs. sum (cheaper).  
- **Upsample**: nearest is fast/stable; bilinear can oversmooth; learned upsample helps edges.  
- **Post-fusion depth**: 1–3 convs; use RepConv at train time and fuse for inference.  
- **Weights (BiFPN)**: enable learnable scalars with ReLU and $\epsilon$-stabilized normalization.

---

## Practical Tips
- **Small objects**: add $N_2$ and/or reduce early downsampling in the backbone; don’t over-shrink channels at $N_3$.  
- **Latency-budgeted**: prefer concat at fewer sites, replace pooling-downs with stride-2 convs, and export fused weights.  
- **Stability**: align channels before fusion; use `SyncBatchNorm` for multi-GPU if micro-batches.  
- **Debugging**: assert spatial sizes match before concat; print shapes at each neck node.

---

## See Also
- [[YOLO Family Architecture]] — overview of Backbone ↔ Neck ↔ Head  
- [[YOLO Backbone (Detailed)]] — feature extractor that feeds the neck  
- [[YOLO Head]] — decoders and losses operating on neck outputs
