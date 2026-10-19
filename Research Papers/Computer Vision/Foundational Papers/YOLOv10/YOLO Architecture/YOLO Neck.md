# YOLO Neck 
#cv/object-detection #yolo/neck  

The **neck** is a *feature blender*. It takes multi-size feature maps from the [[YOLO Backbone (Detailed)]] (small/medium/large) and **mixes** them so each scale has both **detail** and **context** before the [[YOLO Head]] makes predictions.

---

## Inputs → Neck → Outputs (at a glance)

- **Inputs from Backbone:**  
  - Small objects: level $P_3$ (stride $8$)  
  - Medium objects: level $P_4$ (stride $16$)  
  - Large objects: level $P_5$ (stride $32$)

- **Neck mixes them** with:
  - **Upsample** (make a map bigger)
  - **Downsample** (make a map smaller)
  - **Lateral fuse** (combine two maps: *concat* or *sum*)
  - A couple of **Conv–BN–Activation** blocks to clean up features

- **Outputs to Head:**  
  - $N_3$ (small), $N_4$ (medium), $N_5$ (large) — same sizes as inputs, but **richer**.

```
Backbone features      Neck mixes           Output to Head
     P5  (↓) ── up ──┐
     P4  +───────────┼── conv ── down ──┐         N5 (large)
     P3  (↑ from P4) └── conv ──────────┼── conv ─ N4 (medium)
                                 (↑)    └────────── N3 (small)
Legend: up = upsample ×2, down = stride-2 conv/pool, + = fuse (concat/sum)
```

---

## Why do we need a Neck?
- **Backbone top layers** see **big picture** (good for large objects) but lose fine details.  
- **Backbone early layers** keep **edges/textures** (good for small objects) but know less about context.  
- The **neck combines both** so **each scale** is good at its job.

---

## The 4 common neck styles (plain English)

1. **FPN** (Feature Pyramid Network) — *Top-down only*  
   - Take the strong, high-level map ($P_5$), **upsample**, and **mix** into $P_4$, then into $P_3$.  
   - Result: small maps gain **semantics**.

2. **PAN / PAFPN** — *FPN plus a bottom-up pass*  
   - After top-down, go **back up** (bottom-up) so **details** from $N_3$ flow **up** to $N_4$ and $N_5$.  
   - **This is the typical YOLO choice** (often called PAFPN).

3. **BiFPN** — *Bi-directional with learnable mixing weights*  
   - Like PAFPN, but the **blend ratios are learned** (the model decides how much to take from each path).

4. **Light/mobile necks**  
   - Same ideas, but use **depthwise** or **re-parameterized** convs for speed on edge devices.

> **Quick rule of thumb:**  
> - Most YOLO repos use **PAFPN** by default.  
> - If you need extra control/speed, try **BiFPN** (learned weights) or **lighter blocks**.

---

## Minimal mental model (3 steps)

1. **Align channels** with $1{\times}1$ conv (so tensors can mix cleanly).  
2. **Resize then fuse**: upsample/downsample until sizes match, then **concat or sum**, followed by a $3{\times}3$ conv.  
3. **Repeat across levels** so **all three outputs** ($N_3,N_4,N_5$) are ready for the head.

---

## What to tweak (and why)

- **How many levels?**  
  - Standard: $N_3,N_4,N_5$.  
  - Lots of tiny objects? Add **$N_2$** (stride $4$).  
  - Very large inputs? Add **$N_6$/$N_7$**.

- **Fusion type:**  
  - **Concat + conv** = more flexible, slightly heavier.  
  - **Sum** = cheaper, channels must match.

- **Resize ops:**  
  - **Upsample:** nearest is fast/stable; bilinear is smoother.  
  - **Downsample:** stride-2 **$3{\times}3$ conv** is preferred over pooling (keeps features trainable).

- **Width ($C$):**  
  - Same channel count across levels (e.g., $C{=}256$) makes life simpler.  
  - For mobile, shrink $C$ and use depthwise convs.

---

## Tiny pseudocode (read like English)
```python
# Inputs: p3, p4, p5 from the backbone

# 1) Align channels for clean fusion
p3 = conv1x1(p3); p4 = conv1x1(p4); p5 = conv1x1(p5)

# 2) Top-down (FPN)
n4 = conv3x3( concat(p4, up2(p5)) )
n3 = conv3x3( concat(p3, up2(n4)) )

# 3) Bottom-up (PAN)
n4 = conv3x3( concat(n4, down2(n3)) )
n5 = conv3x3( concat(p5, down2(n4)) )

# Outputs to head: n3 (small), n4 (medium), n5 (large)
return n3, n4, n5
```

---

## Quick checklist (when building/debugging)
- [ ] Do the **spatial sizes** match before you concat/sum?  
- [ ] Are **channels aligned** (use $1{\times}1$ conv) before sum?  
- [ ] Are you exporting **fused** (re-param) weights for inference?  
- [ ] For **small objects**, do you have a strong $N_3$ (or consider $N_2$)?  
- [ ] For **latency**, reduce post-fusion conv depth or switch to **sum** fusion.

---

## FAQ

**Q: PAFPN vs. FPN — which should I pick?**  
A: Start with **PAFPN** (it’s standard in YOLO). If you need even more control, try **BiFPN**.

**Q: My concatenation throws a size error.**  
A: Print tensor shapes. **Resize first**, then fuse.

**Q: How do I help tiny objects?**  
A: Add **$N_2$** (stride $4$), keep more channels at the smallest level, and avoid too much early downsampling in the backbone.

---

## See also
- [[YOLO Family Architecture]] — where the neck sits and why  
- [[YOLO Backbone (Detailed)]] — the multi-scale features the neck mixes  
- [[YOLO Head]] — final predictions made per scale
