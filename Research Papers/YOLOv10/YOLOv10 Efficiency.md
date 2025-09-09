# YOLOv10 Efficiency
#yolo/v10 

Yolov10 improves the efficiency of the model architecture through 3 methods:
- [[#Lightweight Classification Head]]
- [[#Decoupled Downsampling]]
- [[#Rank-Guided Block Design]]
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
2. Replace blocks with lowest ranks with cheaper blocks (CIB)
3. Repeat till performance degradation is observed.

### Intrinsic Numerical Rank
The rank is calculated on the last Convolution block of each stage (which are group of blocks like ELAN).

Note that convolution weights have $W \in R^{C_{in} \times C_{out} \times k \times k}$ per stage.
1. Reshape convolution weights into flat matrix $\tilde{W} \in R^{C_{in} \times C_{out} . k^2 }$
2. Perform singular value decomposition (SVD).
   $\hat{W} = U.\Sigma.V^T$ where 
   - $\Sigma = diag(\sigma_{1}, \sigma_{2}, \dots)$ and 
   - $\sigma_{1} > \sigma_{2} > \dots > \sigma_{n}$
3. Let $\lambda_{max}$ be highest singular value ($\sigma_{1}$)
4. Then, let the threshold be half of strongest value.
   If the singular value is half as strong as the largest, we count it.    
   Rank $r$: $\text{no. of }\sigma_{i} > \frac{\lambda_{max}}{2}$
5. Normalize to get no. of useful channels: $\frac{r}{C_{out}}$


### Compact Inverted Blocks (CIB)
![[compact-inverted-block.png]]

Note that for all the explanations below, we will be using a simpler versions of 3 layers instead of actual 5 layers.

YOLOv8 bottleneck blocks use
- `1x1 Conv`: Reduce or mix channels
- `3x3 Conv`: Spatial Mixing (Full conv across all channels)
- `1x1 Conv`: Restore channels
- `Residual Connections`
Cost of dense `3x3 Conv`: $\text{FLOPs} \approx C_{in} \times C_{out} \times 3 \times 3$
Total Cost of `bottleneck`: $C_{in} \times C_{mid} + C_{mid} \times C_{mid} \times 3 \times 3 + C_{mid} \times C_{out} = 2C^2 + 9C^2$

On the other hand, CIB use
- `3x3 Depthwise Conv`: Spatial mixing (also downsample if stride > 1)
- `1x1 Conv`: Channel Mixing (in -> out)
- `3x3 Depthwise Conv`: More Spatial mixing 
- `Residual Connections`
Cost of `3x3 Depthwise Conv`: $\text{FLOPs} \approx C_{mid} \times 3 \times 3$
Total cost of `CIB`: $C_{in} \times 3 \times 3 + C_{in} \times C_{out} + C_{out} \times 3 \times 3 = C^2 + 18C$


### CIB vs bottleneck block
```python
import torch
import torch.nn as nn

def conv_bn_act(in_ch, out_ch, k, s=1, g=1, act=True):
    """Helper: Conv2d + BatchNorm2d + SiLU (optional)."""
    padding = k // 2
    layers = [nn.Conv2d(in_ch, out_ch, k, s, padding, groups=g, bias=False),
              nn.BatchNorm2d(out_ch)]
    if act:
        layers.append(nn.SiLU(inplace=True))
    return nn.Sequential(*layers)

class YoloBottleneck(nn.Module):
    """
    Simplified YOLO bottleneck (used in pre-YOLOv10):
      1x1 conv -> 3x3 conv -> 1x1 conv
    - First 1x1 reduces channels
    - 3x3 mixes spatially
    - Last 1x1 restores channels
    Residual connection if in/out shapes match.
    """
    def __init__(self, in_ch, out_ch, stride=1, hidden_ratio=0.5):
        super().__init__()
        hidden = int(out_ch * hidden_ratio)
        self.down = conv_bn_act(in_ch, hidden, 1)
        self.conv = conv_bn_act(hidden, hidden, 3, s=stride)
        self.up   = conv_bn_act(hidden, out_ch, 1, act=False)
        self.use_res = (stride == 1 and in_ch == out_ch)

    def forward(self, x):
        y = self.down(x)
        y = self.conv(y)
        y = self.up(y)
        if self.use_res:
            y = y + x
        return y

class CIB(nn.Module):
    """
    Compact Inverted Block (YOLOv10):
      DW 3x3 -> PW 1x1 -> DW 3x3
    - First depthwise handles downsampling if stride>1
    - Pointwise mixes channels
    - Second depthwise (optionally large kernel in deep stages) mixes spatially
    Residual connection if in/out shapes match.
    """
    def __init__(self, in_ch, out_ch, stride=1, large_kernel=False):
        super().__init__()
        k2 = 7 if large_kernel else 3
        self.dw1 = conv_bn_act(in_ch, in_ch, 3, s=stride, g=in_ch)  
        self.pw  = conv_bn_act(in_ch, out_ch, 1)
        self.dw2 = conv_bn_act(out_ch, out_ch, k2, g=out_ch)
        self.use_res = (stride == 1 and in_ch == out_ch)

    def forward(self, x):
        y = self.dw1(x)   # depthwise
        y = self.pw(y)    # pointwise
        y = self.dw2(y)   # depthwise
        if self.use_res:
            y = y + x
        return y

# Example usage
if __name__ == "__main__":
    x = torch.randn(1, 64, 80, 80)
    bottleneck = YoloBottleneck(64, 64)
    cib = CIB(64, 64)

    print("YoloBottleneck out:", bottleneck(x).shape)
    print("CIB out:", cib(x).shape)
```
