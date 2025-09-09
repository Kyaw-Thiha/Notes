# YOLOv10 Efficiency
#yolo/v10 
## Lightweight Classification Head
- Classification Head: Predict which class the object belongs to
- Regression Head: Predict box coords (where it is)

It is noted that errors in Regression head have higher effects on the accuracy score.
So, a lightweight version of classification head is used.

### Architecture
In previous YOLO models, a 3x3 convolution layer is used as classification head.
Total Cost: $C_{in} \times C_{out} \times 3 \times 3$

To make it more lightweight
- **Depthwise Convolution**: Separate $3\times3$ kernal for each channel is used 
- **Pointwise Convolution**: After depthwise conv, use $1 \times 1$ conv to mix the channels.

Total Cost: $C_{in} \times 3 \times 3 + C_{in}.C_{out}$

### Standard 3x3 Convolution
Every output channel looks at all input channels, each with its own 3×3 kernel, then sums them.
```
Input (3 channels)      3×3 Conv Filter Bank        Output (2 channels)
 ┌─────────┐             ┌─────────────┐            ┌─────────┐
 │ chan1   │─┐           │ filter for  │── sum ────►│ out1    │
 ├─────────┤ ├─► all ──► │   out1      │            ├─────────┤
 │ chan2   │─┤   mixed   ├─────────────┤            │ out2    │
 ├─────────┤ ┘           │ filter for  │── sum ────►└─────────┘
 │ chan3   │────────────►│   out2      │
 └─────────┘             └─────────────┘
```

### Depthwise Separable 3x3 Convolution
**Step A: Depthwise (per-channel 3×3)**  
Each channel filtered independently — no mixing yet.
**Step B: Pointwise (1×1 mixing)**  
Now a 1×1 conv linearly combines the depthwise outputs to produce new channels.
```
Step A: Depthwise 3×3 (per-channel, no mixing)

 Input channels     After depthwise conv
 ┌─────────┐        ┌─────────┐
 │ chan1   │─3×3───►│ chan1'  │
 ├─────────┤        ├─────────┤
 │ chan2   │─3×3───►│ chan2'  │
 ├─────────┤        ├─────────┤
 │ chan3   │─3×3───►│ chan3'  │
 └─────────┘        └─────────┘


Step B: Pointwise 1×1 (mix channels)

 Depthwise out      1×1 conv mix      Final out
 ┌─────────┐        ┌─────────┐       ┌─────────┐
 │ chan1'  │─┐      │         │ ────► │ out1    │
 ├─────────┤ ├─────►│  1×1    │ ────► ├─────────┤
 │ chan2'  │─┤      │  mixing │       │ out2    │
 ├─────────┤ ┘      │         │       └─────────┘
 │ chan3'  │───────►│         │
 └─────────┘        └─────────┘

```

Essentially, pointwise linear mixing is more efficient than 'hidden' mixing inside standard convolution, with relatively small hit on performance.

## Spatial-Channel Decoupled Downsampling
Previous YOLO models often use $3 \times 3$ convolution with stride of 2.
This carries out the downsampling process by
- Halving spatial resolution: $H \times W \to \frac{H}{2} \times \frac{W}{2}$
- Doubling channels: $C \to 2C$


### Standard Conv 3x3 with Stride-2
```
Input Feature Map (H × W × C)

 ┌───────────────┐
 │               │
 │   [3×3 conv]  │   stride=2, output channels=2C
 │   mixes ALL   │───►  Output Feature Map (H/2 × W/2 × 2C)
 │   channels    │
 │   + downsamp  │
 └───────────────┘

Cost ~ O(18 C²) params, O((9/2) HWC²) FLOPs
```
Mixes channels & downsamples spatially in one heavy op.
Cost: $C_{in} \times C_{out} \times 3 \times 3 = C \times 2C \times 9 = 18.C^2$

### Decoupled Downsampling
Step-A: pointwise 1×1 (channel mixing)
```
Input Feature Map (H × W × C)
        │
        ▼
 ┌───────────────┐
 │   1×1 conv    │   (cheap channel mixing)
 │   C → 2C      │───►  Intermediate (H × W × 2C)
 └───────────────┘
```
Cost: $C \times 2C = 2C^2$


Step-B: depthwise 3×3 with stride=2 (spatial downsampling)
```
Intermediate (H × W × 2C)
        │
        ▼
 ┌───────────────┐
 │ Depthwise 3×3 │   (one filter per channel, stride=2)
 │   stride=2    │───►  Output Feature Map (H/2 × W/2 × 2C)
 └───────────────┘
```
Cost: $2C \times 9 = 18C$

Total Cost: $2C^2 + 18C$

## Rank-Guided Block Design
1. Rank blocks inside the model in order of performance
2. Replace blocks with lowest ranks with cheaper blocks
3. Repeat till performance degradation is observed.
