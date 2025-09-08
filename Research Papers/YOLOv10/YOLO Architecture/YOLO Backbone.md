# YOLO Backbone (Simple Guide)
#cv/object-detection #yolo/backbone

**One sentence:** The **backbone** is YOLO’s *feature extractor*. It converts the input image into a few **feature maps at different sizes** (small/medium/large) that the [[YOLO Neck]] mixes and the [[YOLO Head]] uses to predict boxes/classes.

---

## What comes in, what goes out

- **Input:** RGB image (e.g., `640×640×3`)
- **Output:** 3 (sometimes 4) feature maps at fixed **strides**:
  - **$P_3$** (stride $8$) → small objects *(80×80 for 640 input)*
  - **$P_4$** (stride $16$) → medium objects *(40×40)*
  - **$P_5$** (stride $32$) → large objects *(20×20)*
  - *(optional)* **$P_2$** (stride $4$) for **tiny** objects

```
Image (640x640)
   ↓ Stem (downsample ×2)
   ↓ Stage 2 (×2) → P3 (80x80, s=8)
   ↓ Stage 3 (×2) → P4 (40x40, s=16)
   ↓ Stage 4 (×2) → P5 (20x20, s=32)
   → SPPF (adds multi-scale context at the top)
```

> Rule of thumb: after $n$ downsamplings by $×2$, the **stride** is $s = 2^n$.

---

## What the backbone is made of 

- **Stem:** a quick size reduction with a stride-2 conv (older YOLOs sometimes use Focus/Space-to-Depth).
- **Stages:** repeated **Conv → BN → Activation** blocks with *skip/partial* connections to make training stable.
- **SPP/SPPF:** a tiny block at the end that mixes **context at multiple kernel sizes** (cheap and helpful).

**Common block names you’ll see**
- **Residual / Bottleneck:** classic skip connections.
- **CSP / C2f:** “split, process one side, then fuse” — efficient, strong gradients.
- **ELAN / E-ELAN:** aggregation pattern popularized around YOLOv7.
- **RepConv:** multi-branch at train time, **fused** to a single conv at inference.

---

## YOLO versions in one glance

| Family | Backbone idea (simple) |
|---|---|
| **YOLOv3** | **Darknet-53** (residual stacks) |
| **YOLOv4 / v5** | **CSPDarknet** + **SPP/SPPF** |
| **YOLOv7** | **E-ELAN** + re-param tricks |
| **YOLOv8** | **C2f** modules + **SPPF** |

> Don’t get stuck on names — the **outputs ($P_3,P_4,P_5$), strides, and channel sizes** matter most.

---

## How to think about design choices

- **Levels to output:** Usually $P_3,P_4,P_5$.  
  - Lots of tiny objects? add **$P_2$ (stride 4)**.
- **Depth/width (n/s/m/l/x):** Bigger model = more blocks/channels.
- **Activation:** **SiLU** is a good default; **ReLU6** for mobile; **LeakyReLU** in older configs.
- **Normalization:** **BatchNorm** (often SyncBN for multi-GPU).
- **SPPF:** keep it **on** — cheap, usually helps.

---

## Tiny pseudocode (read like English)
```python
# x: input image
x = ConvBNAct(x, k=3, s=2)        # stem (↓2)

x = ConvBNAct(x, k=3, s=2); x = C2fStack(x)  # stage 2 (↓4) → P3 candidate
p3 = x

x = ConvBNAct(x, k=3, s=2); x = C2fStack(x)  # stage 3 (↓8) → P4 candidate
p4 = x

x = ConvBNAct(x, k=3, s=2); x = C2fStack(x)  # stage 4 (↓16) → top features
x = SPPF(x)                                  # context mix
p5 = x                                       # (↓32)

return p3, p4, p5
```

---

## Quick checklist (when building/debugging)

- [ ] Do you emit **$P_3,P_4,P_5$** at strides **8/16/32**?  
- [ ] Are **channels** reasonable/consistent so the [[YOLO Neck]] can fuse them (e.g., all 256)?  
- [ ] **SPPF** included at the top?  
- [ ] For **tiny objects**, consider adding **$P_2$** (stride 4) or reducing early downsampling.  
- [ ] Export **re-parameterized** (fused) convs for deployment if using Rep* blocks.

---

## See also
- [[YOLO Family Architecture]] — how Backbone, Neck, Head fit together  
- [[YOLO Neck]] — feature mixing/fusion between scales  
- [[YOLO Head]] — prediction layers and losses
