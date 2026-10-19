# ResNet Variants

![ResNet Variants|500](https://notes-media.kthiha.com/ResNet-Variants/ed4b2adf859d701df141c4aed228cfa5.png)

Common structure across all variants:
- `conv1`: $7 \times 7$, $64$ filters, stride $2$
- `conv2_x`: $56 \times 56$ feature map size
- `conv3_x`: $28 \times 28$ feature map size  
- `conv4_x`: $14 \times 14$ feature map size
- `conv5_x`: $7 \times 7$ feature map size
- `Final`: Global average pooling → $1000\text{-d}$ fc → softmax

The `number after conv` indicates the stage.
`_x` indicates multiple blocks.

---
## Building Blocks

### Basic Block (ResNet-18/34)
Used in shallower networks (18 and 34 layers)
```
[3×3, 64]
[3×3, 64]  × n times
```

Two $3 \times 3$ [[Convolution Layer]] per block
- First layer: `3×3 conv`
- Second layer: `3×3 conv`
- Shortcut connection wraps both layers

---
### Bottleneck Block (ResNet-50/101/152)
Used in deeper networks for efficiency
```
[1×1, 64 ]
[3×3, 64 ]  × n times
[1×1, 256]
```

Three layers per block:
1. `1×1 conv`: Reduces dimensions (bottleneck)
2. `3×3 conv`: Main computation with smaller input/output dims
3. `1×1 conv`: Restores dimensions (expanding)

**Why bottleneck?** 
- Reduces `time complexity` compared to basic block
- The $1 \times 1 \text{ layers}$ `reduce` then `restore` dimensions
- $3 \times 3 \text{ layers}$ operates on `smaller feature maps`
- Parameter-free identity shortcuts particularly important here
- If identity shortcut replaced with projection, time complexity and model size doubled 
  (shortcut connects to high-dimensional ends)

---
## Usage Rule of Thumb

`ResNet-18/34`: 
- Faster training and inference
- Limited computational resources
- Basic block architecture

`ResNet-50`: 
- Sweet spot for most applications
- Good accuracy/efficiency trade-off
- Most commonly used for transfer learning

`ResNet-101/152`: 
- Maximum accuracy needed
- Sufficient computational resources
- Diminishing returns vs ResNet-50

---