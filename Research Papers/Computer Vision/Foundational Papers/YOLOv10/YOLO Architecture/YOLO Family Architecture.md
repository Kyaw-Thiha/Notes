# YOLO Family Architecture 
#cv/object-detection #yolo/architecture

All YOLO models follow the same big idea — a **Backbone** extracts multi-scale features → a **Neck** fuses them → a **Head** predicts boxes, objectness, and classes → optional **post-processing** (e.g., NMS).

Reference: https://viso.ai/computer-vision/yolo-explained/

---

## Big Picture

```
Image
  └─ Backbone → P3,P4,P5  (multi-scale features; strides 8/16/32)
      └─ Neck → N3,N4,N5  (fused features at same strides)
          └─ Head → boxes + objectness + class scores
              └─ Post-process → NMS (classic) or NMS-free (some modern YOLO)
```

- **Backbone**: CNN feature extractor (e.g., Darknet/CSP/C2f/ELAN); often ends with SPP/SPPF.  
- **Neck**: top-down & bottom-up fusion (FPN/PAN/PAFPN/BiFPN).  
- **Head**: decoupled towers for **reg/objectness** and **classification**; **anchor-based** (v3–v5/7) or **anchor-free** (YOLOX, v8+).  
- **Post-processing**: classic **NMS** variants; newer designs (e.g., **YOLOv10**) can train for **NMS-free** inference.

> Typical shapes for 640×640 input:  
> $P_3/N_3:\ 80\times80$ (stride 8), $P_4/N_4:\ 40\times40$ (stride 16), $P_5/N_5:\ 20\times20$ (stride 32).

---

## Anchor-Based vs Anchor-Free (quickly)
- **Anchor-based**: predict offsets from preset box sizes per cell (needs anchor matching in training).  
- **Anchor-free**: predict centers+sizes or 4 edge distances directly (simpler assignment, robust scaling).

**Label assignment** (who is positive?): static IoU/center rules → dynamic assigners (e.g., **SimOTA/TAL**) that pick top-k by quality.

---

## What Changed Across YOLO Versions (very concise)

| Era | Backbone | Neck | Head | Notes |
|---|---|---|---|---|
| **v1 (2015/16)** | Conv + FC | – | Coupled | First one-stage grid-based detector. |
| **v2 (YOLO9000)** | Darknet-19 | – | Anchor-based | BatchNorm, anchors, multi-scale training. |
| **v3 (2018)** | **Darknet-53** | FPN-like | Anchor-based, multi-scale | Three prediction scales. |
| **v4 (2020)** | **CSPDarknet-53** | **SPP + PAN** | Anchor-based | “Bag of Freebies/Specials”, large practical boost. |
| **v5** | CSP-style + **SPPF** | PAFPN | Anchor-based (popular, pragmatic) | Strong tooling & exports. |
| **YOLOX** | CSP-style | PAN | **Anchor-free, decoupled** | Modern training & assigners. |
| **v6** | EfficientRep | Rep-PAN | **Efficient decoupled** | TAL, industrial/edge focus. |
| **v7** | **E-ELAN** | PAFPN | Anchor-based | Re-parameterization, training refinements. |
| **v8** | **C2f + SPPF** | PAFPN | **Anchor-free, decoupled** | Ultralytics’ current workhorse. |
| **v9** | **GELAN** | – | – | **PGI** for better gradient/info flow. |
| **v10** | Holistic tweaks | – | **NMS-free** (dual assignments) | Optimized for latency + accuracy. |
| **v11** | **C3k2**, **C2PSA** | – | – | Efficiency and attention refinements. |

*(Dashes “–” mean “same idea family / not the main novelty.”)*

---

## Minimal Mental Model (how to design one)
1. **Choose output strides** (usually 8/16/32; add 4 for tiny objects).  
2. **Pick backbone motif** (CSP/C2f/ELAN/Rep) + **SPPF on** by default.  
3. **Pick neck** (start with **PAFPN**; BiFPN if you want learned fusion weights).  
4. **Pick head** (**decoupled + anchor-free** is a great default).  
5. **Label assignment** (start with **SimOTA/TAL** style).  
6. **Losses**: IoU-family for box (CIoU/EIoU), BCE/Focal for cls/obj, optional DFL for finer boxes.  
7. **Post-proc**: NMS or **NMS-free** pipeline (if your head/assignment supports it).

---

## Tiny E2E Pseudocode (read like English)
```python
# x: image → backbone → neck → head → (optional) post-processing
p3,p4,p5 = backbone(x)            # multi-scale features
n3,n4,n5 = neck(p3,p4,p5)         # fused features
preds    = head([n3,n4,n5])       # per-scale: box(4), obj(1), cls(C)
detections = nms_or_nms_free(preds)
```

---

## See Also 
- [[YOLO Backbone]] — how features are extracted  
- [[YOLO Neck]] — how features are fused  
- [[YOLO Head]] — how predictions are made  
- [[Non-Maximum Suppression (NMS)]] — classic duplicate removal
- Reference: https://viso.ai/computer-vision/yolo-explained/
