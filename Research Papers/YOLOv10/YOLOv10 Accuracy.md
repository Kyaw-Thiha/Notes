# YOLOv10 Accuracy
#yolo/v10/accuracy 

Yolov10 improve the accuracy of YOLOv8 through 
- [[#Large Kernal Convolution]]
- [[#Partial Self-Attention]]

---
## Large Kernal Convolution
![[compact-inverted-block.png]]

Replace the second `3x3 depthwise conv` with `7x7 depthwise conv` inside [[YOLOv10 Efficiency#Compact Inverted Blocks (CIB)|CIB]] blocks.

`7x7 depthwise conv` has big receptive field, so it is good for accuracy, but harder to optimize during training.
To help with optimization, a `3x3 depthwise conv branch` is used during training before being merged to the `7x7 depthwise conv` during inference.
```
Training:  Input ──► [7×7 DW Conv] ──┐
                                     ├─► Sum ─► Output
           Input ──► [3×3 DW Conv]───┘

                 (Reparameterization step)
                 Embed 3×3 into 7×7:
                 ┌───────┐    ┌───────┐    ┌───────┐
   Original W7:  │■■■■■■■│    │0000000│    │■■■■■■■│
                 │■■■■■■■│    │0000000│    │■■■■■■■│
                 │■■■■■■■│ +  │00aaa00│ =  │■■aaa■■│
                 │■■■■■■■│    │00bbb00│    │■■bbb■■│
                 │■■■■■■■│    │00ccc00│    │■■ccc■■│
                 │■■■■■■■│    │0000000│    │■■■■■■■│
                 │■■■■■■■│    │0000000│    │■■■■■■■│
                 └───────┘    └───────┘    └───────┘
                    W7         pad(W3)         W7'

Inference: Input ──► [7×7 DW Conv'] ──► Output
```

- This `7x7 depthwise conv` can contaminate small object detection + I/O overhead, so it is only employed within deep stages.
- Only smaller model variants (Nano & Small) employ this since larger models already have big receptive field (so lower benefit).

Code example can be see in the [[YOLOv10 Efficiency#CIB vs bottleneck code|CIB Code Example]].

## Partial Self-Attention
![[partial-self-attention.png]]
`Self-Attention` blocks can help improve accuracy but also has high computational complexity.

To tackle this,
1. The output from the `1x1 Conv` is partitioned into two parts.
2. One of the part is put through the `MHSA + FFN` self-attention block.
3. The 2 parts are then `concatenated` together, and fused together by `1x1 Conv`.

```
Input: (B, H, W, C)
	        │
	        ▼
	   1×1 Convolution
	        │
	        ▼
	   (B, H, W, C')
	        │
	   Split channels
	        │
   ┌────────────────┐
   │                │               
   ▼                ▼               
(B,H,W,C'/2)   (B,H,W,C'/2)   ← even half/half split
   │                │
[ MHSA + FFN ]   (Identity)
   │                │
   ▼                ▼
(B,H,W,C'/2)   (B,H,W,C'/2)
   └────────────────┘
			│
			▼
		Concatenate
			│
			▼
	   (B, H, W, C')
			│
	  1×1 Convolution
			│
			▼
	   Final Output
```

To further optimize
- Make `dim(Q) = dim(K) = d_model/2` and keep `dim(V) = d_model` inside the `MHSA`
  Normally, `dim(Q) = dim(K) = dim(V) = d_model`
- Replace `LayerNorm` with `BatchNorm` inside the transformer block.
- `PSA` is only applied in `stage-4` (which is deepest stage in backbone) by which time `HxW` has undergone max downsampling.
  Intuitively, YOLO `convolutions` help learn local patterns while the `self-attention` help learn global patterns.
  Putting it in deepest block is akin to learning from thumbnail instead of original high-res image.

## See Also
- [[YOLOv10]]
- [[YOLOv10 Efficiency]]
- [[Dual Label Assignment]]
