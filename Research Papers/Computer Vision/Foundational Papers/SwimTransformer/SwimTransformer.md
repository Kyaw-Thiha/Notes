# SwimTransformer
`SwimTransformer` is a hierarchical vision [[Transformer]] that uses shifted windows for self-attention computation.

![SwimTransformer|400](https://i.ytimg.com/vi/z_8lajPxGQo/maxresdefault.jpg?sqp=-oaymwEmCIAKENAF8quKqQMa8AEB-AH-CYAC0AWKAgwIABABGEogWShlMA8=&rs=AOn4CLBQ_T22PCgPGF-THxppfj65Vjp25Q)

This achieve linear complexity with respect to image size.
It also serve as a general-purpose backbone for computer vision tasks. 

---
## Background Info
The key challenges in adapting [[Transformer]] to `Computer Vision` are:
- **Scale Variation**
  Visual entities vary dramatically in scale
  unlike fixed-size word tokens
- **High Resolution**
  Images have more pixels than text has words.
  This makes global attention computationally expensive.

The prior attempts to solve this involve
- `ViT` applying [[Transformer]] to fixed-size image patches.
  But this 
	  - require massive datasets.
	  - produces only single-resolution features.
- Other works use [[Self-Attention]] in sliding windows.
  But these are inefficient in hardware.

---
## Uniqueness from Prior Research
**Compared to Vision Transformers (`ViT`, `DeiT`)**
![Swim vs ViT|300](https://www.researchgate.net/publication/362908522/figure/fig4/AS:11431281080743494@1661398122382/Comparison-of-architectures-for-Swin-vs-ViT-Xie-et-al-2021-The-ViT-transformer-works.png)
- **Scale**
  `Swim` produces multi-scale feature maps, instead of single-scale feature map.
  This make it suitable for dense prediction tasks
- **Complexity**
  `Swim` achieves $O(n)$ complexity through local windowed attention, instead of `ViT`'s $O(n^2)$.
- **Performance**: Better


**Compared to Other Transformers**
- `Sliding window` approaches suffer from poor hardware efficiency
  `Shifted windows` enable efficient batch computation.
- More efficient than concurrent works while achieving better accuracy-speed trade-offs

---
## Architecture
![SwimTransformer Architecture](https://user-images.githubusercontent.com/26739999/142576715-14668c6b-5cb8-4de8-ac51-419fae773c90.png)

`Swim Transformer` is made of a four-stage hierarchical  design:
1. **Patch Partition**
   Split input into $4 \times 4$ non-overlapping patches
2. **Stage-1**
   Linear embedding to dimension $C$.
   Then, [[#Swim Transformer Block]] blocks at $(\frac{H}{4} \times \frac{W}{4})$
3. **Stage-2 to 4**
   Patch merging layer $(2 \times 2 \text{ downsampling})$
   Then, [[#Swim Transformer Block]] blocks at
	   - **Stage-2**: $\left( \frac{H}{8}, \frac{W}{8} \right)$, dimension $2C$
	   - **Stage-3**: $\left( \frac{H}{16}, \frac{W}{16} \right)$, dimension $4C$
	   - **Stage-4**: $\left( \frac{H}{32}, \frac{W}{32} \right)$, dimension $8C$
	   
### Shifted Window 
![Shifted Window|500](https://substackcdn.com/image/fetch/$s_!cl_l!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff9719e77-242c-44ac-8b3f-fef199a42d91_1521x870.png)
**Window-based Self-Attention** ($\text{W-MSA}$)
- Partition image into $M \times M$ non-overlapping windows
  $(M=7 \text{ by default})$
- Compute self-attention only within each window.
- Linear Complexity: $O(4hwC^2 + 2M^2hwC)$

**Shifted-Window Self-Attention** $(\text{SW-MSA})$
- Alternates between regular and shifted window partitioning across consecutive layers
- Shift windows by $\left( \frac{M}{2}, \frac{M}{2} \right)$ pixels
- Creates cross-window connections while maintaining efficiency

### Swim Transformer Block
`Swin Transformer` is built by replacing the standard multi-head self attention module by [[#Shifted Window]].

For **two** consecutive swim transformer blocks,
1. Apply [[Self-Attention|multi-head self-attention]] on $\text{W-MSA}$
   $\hat{z}^t = \text{W-MSA}( \ \text{LN}(z^{t-1}) \ )+ z^{t-1}$
2. Apply the [[Neural Network|FFN]].
   $z^t = \text{MLP}( \ \text{LN}(\hat{z}^t) \ ) + \hat{z}^t$ 
3. In next block, apply [[Self-Attention|multi-head self-attention]] on $\text{SW-MSA}$
   $\hat{z}^{(t+1)} = \text{SW-MSA}( \ \text{LN}(z^{t}) \ )+ z^{t}$
4. Apply the [[Neural Network|FFN]].
   $z^{t+1} = \text{MLP}( \ \text{LN}(\hat{z}^{t+1}) \ ) + \hat{z}^{t+1}$ 

### Relative Position Bias
In [[Self-Attention|standard self-attention]], 
$$
\text{Attention}(Q, K, V)
= \text{SoftMax}\left( 
\frac{Q K^T}{\sqrt{ d }} \right) \cdot V
$$
The transformer doesn't have spatial information of the patches.
Two patches could be adjacent or far apart, and the attention mechanism wouldn't know the difference.

In order to solve this, we add a bias matrix
$$
\text{Attention}(Q, K , V)
= \text{SoftMax}\left( 
\frac{QK^T}{\sqrt{ d }} + B \right) \cdot V
$$
where $B \in \mathbb{R}^{M^2 \times M^2}$ is a `learnable bias matrix`

#### Parameterization Trick
Consider two patches at positions $(i, j)$ and $(k, l)$ in an $M \times M$ window.
Instead of learning $M^2$ parameters, we could instead learn their relative positions:
- $\Delta x = i - k$ (ranges from $-(M+1)$ to $+ (M-1)$)
- $\Delta y = j - l$ (ranges from $-(M+1)$ to $+ (M-1)$)

From $-(M+1)$ to $+(M-1)$, we have $2M - 1$ possible values.

**How it works**
1. **Learn $\hat{B}$**
   A $(2M-1) \times (2M-1)$ matrix indexed by relative positions
2. **Index into $\hat{B}$**: 
   For each pair of patches $(i,j)$ and $(k,l)$:
    - Compute relative position: $(\Delta x, \Delta y) = (i-k, j-l)$
    - Look up bias value: $B[i,j,k,l] = \hat{B}[\Delta x + M-1, \Delta y + M-1]$
3. **Apply to attention** 
   Add these bias values to $QK^T$ before softmax

---
## Architecture Variants 
- **Swin-T**: C$=96$, layers=$\{ 2, 2, 6, 2 \}$ 
  (~29M params, similar to ResNet-50)
- **Swin-S**: C$=96$, layers=$\{ 2, 2, 18, 2 \}$
  (~50M params, similar to ResNet-101)
- **Swin-B**: C$= 128$, layers=$\{ 2, 2, 18, 2 \}$
  (~88M params, similar to ViT-B)
- **Swin-L**: C$= 192$, layers=$\{ 2, 2, 18, 2 \}$
  (~197M params)

---
## See Also
- [Paper](https://arxiv.org/abs/2103.14030)