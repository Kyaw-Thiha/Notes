# YOLOv10 Efficiency
#yolo/v10/efficiency 

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

```
Prev. YOLO Head (e.g. YOLOv8)
------------------------------
   [ Input Features ]
          |
     +----+----+
     |         |
 [ Classification Head ]             [ Regression Head ]
   1×1 Conv (dense, channel mix)       1×1 Conv (dense)
   3×3 Conv (dense, spatial + chan)    3×3 Conv (dense)
   1×1 Conv (dense, channel mix)       1×1 Conv (dense)
   FLOPs ~5.95G, Params ~1.51M         FLOPs ~2.34G, Params ~0.64M
          |                                   |
   [ Class Scores ]                     [ Box Offsets ]


YOLOv10 Lightweight Head
------------------------
   [ Input Features ]
          |
     +----+----+
     |         |
 [ Classification Head ]             [ Regression Head ]
   3×3 DW Conv (spatial only)          1×1 Conv (dense)
   3×3 DW Conv (spatial only)          3×3 Conv (dense)
   1×1 PW Conv (channel mix)           1×1 Conv (dense)
   (depthwise separable stack)         (kept strong, dense convs)
   Low FLOPs / Params                  Higher FLOPs / Params
          |                                   |
   [ Class Scores ]                     [ Box Offsets ]
```

## Decoupled Downsampling
```
Prev. YOLO Downsampling (Standard)
----------------------------------
   [ Input: H × W × C ]
            |
     3×3 Conv, stride=2 (dense)
       - halves spatial size (H/2 × W/2)
       - doubles channels (2C)
       - spatial + channel mixed together
       Cost: O( (9/2) · HWC² ), Params: O(18C²)
            |
   [ Output: H/2 × W/2 × 2C ]


YOLOv10 Decoupled Downsampling
------------------------------
   [ Input: H × W × C ]
            |
   1×1 PW Conv (dense, channel mix)
       - adjust channels C → 2C
       - cheap (no spatial context)
            |
   3×3 DW Conv, stride=2 (depthwise)
       - downsample H×W → H/2 × W/2
       - spatial only, per-channel
       - very cheap compared to dense 3×3
       Cost: O(2HWC² + (9/2)HWC), Params: O(2C² + 18C)
            |
   [ Output: H/2 × W/2 × 2C ]

```

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


### CIB vs bottleneck code
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

## Note about understanding channels
- Channels at the start and end of the model is `3 for RGB`  and `1 for greyscale`.
- Inside the network, it is equal to `H x W x no. of neurons in the layer`. 
  `HxW` maybe downsampled to `H/2 x W/2`.

## See Also
- [[YOLOv10]]
- [[YOLOv10 Accuracy]]
- [[Dual Label Assignment]]
