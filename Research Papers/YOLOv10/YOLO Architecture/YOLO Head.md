# YOLO Head (Simple Guide)
#cv/object-detection #yolo/head

**One sentence:** The **head** turns the fused features from the [[YOLO Neck]] into final **box coordinates**, **objectness**, and **class scores** for each location/scale.

---
## Inputs → Head → Outputs (at a glance)

- **Inputs (from Neck):** $N_3, N_4, N_5$ feature maps (strides $8,16,32$).
- **Per location** on each map, the head predicts:
  - **Box** (4 numbers)
  - **Objectness** (1 number = “is there any object?”)
  - **Classes** ($C$ numbers = “which class?”)

> Example shapes (per level):  
> - **Anchor-free:** $(H \times W) \times (4 + 1 + C)$  
> - **Anchor-based (k anchors):** $(H \times W \times k) \times (4 + 1 + C)$

---
## Two big design choices

### 1) **Coupled vs. Decoupled**
- **Coupled** head: one tower predicts box + obj + class together. (Older style.)
- **Decoupled** head (**common now**): **separate small towers** for **box/objectness** and **classification** → easier optimization, faster convergence.

### 2) **Anchor-based vs. Anchor-free**
- **Anchor-based (YOLOv3/v4/v5/v7)**  
  - Predefine $k$ **anchor boxes** (sizes) per cell.  
  - Predict **offsets** $(t_x,t_y,t_w,t_h)$ relative to the anchor, plus objectness and class.  
  - Needs **anchor matching** during training.
- **Anchor-free (YOLOX/YOLOv8)**  
  - No anchors; predict either **center + size** or **distances** to the four sides $(d_l,d_t,d_r,d_b)$.  
  - Simpler label assignment; often more robust to scale/shape.

> Quick pick: If you’re starting fresh, **decoupled + anchor-free** is a great default.

---
## What the head looks like (plain English)

- Take each input feature map ($N_3,N_4,N_5$).
- Run **two small conv stacks**:
  - **Reg/Obj tower** → outputs 4 box numbers + 1 objectness.
  - **Cls tower** → outputs $C$ class logits.
- Repeat for all pyramid levels, then **concatenate** predictions from all levels.

```
       N3 ──► [Reg/Obj tower] ─► box+obj
          └─► [   Cls tower ] ─► class scores

       N4 ──► [Reg/Obj tower] ─► box+obj
          └─► [   Cls tower ] ─► class scores

       N5 ──► [Reg/Obj tower] ─► box+obj
          └─► [   Cls tower ] ─► class scores
```

---

## Losses & training (high level)

- **Box loss:** IoU-style losses (GIoU/DIoU/CIoU/EIoU).  
  - Some heads use **Distribution Focal Loss (DFL)** to predict a discrete distribution for each side/coordinate (smoother, more accurate).
- **Objectness loss:** Binary cross-entropy, sometimes weighted by IoU with matched GT.
- **Classification loss:** BCE with logits or Focal Loss; one-vs-all (sigmoid) is common in YOLO.

**Label assignment (who is positive?)**
- **Static rules:** e.g., closest anchors/centers with IoU thresholds.
- **Dynamic rules:** **OTA/SimOTA/TAL** pick top-$k$ candidates by a combined quality (classification × IoU), often more stable.

---
## Post-processing
- Convert head outputs to **absolute boxes** in image space.
- Apply **confidence filtering** (objectness × class score).
- Run [[Non-Maximum Suppression|NMS]] (or Soft-NMS/DIoU-NMS) to remove duplicates.
- Return final detections: $(x_1,y_1,x_2,y_2,\text{score},\text{class})$.

---
## Minimal pseudocode (decoupled, anchor-free, PyTorch-like)
```python
class YOLOHead(nn.Module):
    def __init__(self, in_chs=(256,256,256), num_classes=80, reg_ch=128, cls_ch=128):
        super().__init__()
        def tower(c_in, c_mid, n=2):
            return nn.Sequential(*[nn.Sequential(
                nn.Conv2d(c_in if i==0 else c_mid, c_mid, 3, 1, 1, bias=False),
                nn.BatchNorm2d(c_mid),
                nn.SiLU(inplace=True)) for i in range(n)])
        self.reg_towers = nn.ModuleList([tower(c, reg_ch) for c in in_chs])
        self.cls_towers = nn.ModuleList([tower(c, cls_ch) for c in in_chs])
        self.reg_heads  = nn.ModuleList([nn.Conv2d(reg_ch, 4, 1)   for _ in in_chs])  # box
        self.obj_heads  = nn.ModuleList([nn.Conv2d(reg_ch, 1, 1)   for _ in in_chs])  # objectness
        self.cls_heads  = nn.ModuleList([nn.Conv2d(cls_ch, num_classes, 1) for _ in in_chs])
    def forward(self, feats):  # feats = [N3, N4, N5]
        outs = []
        for i, f in enumerate(feats):
            r = self.reg_towers[i](f);  b = self.reg_heads[i](r); o = self.obj_heads[i](r)
            c = self.cls_heads[i](self.cls_towers[i](f))
            outs.append((b, o, c))
        return outs  # list over levels: (box, obj, cls)
```

---

## Knobs you’ll actually tune

- **Classes $C$** (dataset dependent).  
- **Head width/depth** (number of channels/blocks in the towers).  
- **Anchor settings** (only if anchor-based): number/sizes per level.  
- **Loss mix:** IoU loss type, DFL on/off, focal loss $\gamma$.  
- **Conf/NMS thresholds:** balance precision vs. recall.  
- **Which levels to predict on:** $(N_3,N_4,N_5)$, maybe add **$N_2$** for tiny objects or **$N_6$** for very large.

---

## Quick checklist (build/debug)

- [ ] Shapes match: box has 4, obj has 1, cls has $C$ per location.  
- [ ] Decoupled towers are **small but not tiny** (2–3 convs is common).  
- [ ] Losses wired correctly (no sigmoid twice, correct target scaling).  
- [ ] Proper label assignment (dynamic assigners reduce manual thresholds).  
- [ ] Post-processing uses sensible **conf/NMS** settings.

---

## See also
- [[YOLO Family Architecture]] — big picture of Backbone ↔ Neck ↔ Head  
- [[YOLO Backbone|YOLO Backbone]] — multi-scale features  
- [[YOLO Neck|YOLO Neck]] — blends features across scales  
- [[Non-Maximum Suppression]] — removes duplicate boxes
