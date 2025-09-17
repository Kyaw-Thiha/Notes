# DETR Architecture
#cv/object-detection/models/detr

![Architecture](https://viso.ai/wp-content/uploads/2024/02/DETR-Architecture.jpg)

DETR is made of three main components:
- CNN backbone
- Encoder-Decoder Transformer
- FFN Prediction Heads

## CNN Backbone
`ResNet` is used as a backbone to extract features from the image.
- `ResNet50` is used for `DETR`
- `ResNet101` is used for `DETR-R101`

![ResNet Architecture](https://www.researchgate.net/publication/349646156/figure/fig4/AS:11431281387544643@1745122801321/The-architecture-of-ResNet-50-vd-a-Stem-block-b-Stage1-Block1-c-Stage1-Block2.png)

- Input image $x \in R^{3 \times W_{0} \times H_{0}}$ is processed with a `ResNet`.
- `ResNet` returns an output feature map $f \in R^{C \times W \times H}$, where
  - $C$ is typically $2048$
  - $H = \frac{H_{0}}{32}$
  - $W = \frac{W_{0}}{32}$
- A `1x1 Conv` is used to reduce channels from $2048$ to $d$ where $d=256$ is the hidden dimension for the Transformer
- The spatial dimension $H \times W$ is flattened into length of$H.W$
- Hence, [[Positional Encoding]] are added since transformer will not know the spatial order otherwise.

### Visualization of ResNet Feature Map
![ResNet Feature Map](https://www.researchgate.net/publication/339315932/figure/fig2/AS:859739332292608@1581989238021/sualization-of-the-features-from-ResNet-a-Input-frame-b-e-Features-are-extracted-on.png)

A good way to visualize `ResNet` and similar `CNN` models is that
- In earlier layers, they focus more on details (local features)
- In later layers, they focus more on global features 

---
## Transformer Encoder
The encoder is a stack of $N = 6$ encoder layers.

Each layer consist of 
1. A [[Self-Attention]] layer
   $x' = \text{LayerNorm}(x + \text{SelfAttention(x)})$
2. A [[Feed-Forward Neural Network]] layer
   Standard `2-Layer MLP` with `RELU` in between.
   $y = \text{LayerNorm(x' + FFN(x'))}$

As shown above, each of the `Self-Attention` and `Feed Forward Network` are wrapped in [[Skip Connection|Residual]] and layer normalization.

![Transformer Encoder](https://www.researchgate.net/publication/334288604/figure/fig1/AS:778232232148992@1562556431066/The-Transformer-encoder-structure.ppm)

For the `Self-Attention`, 
- the queries $Q$, keys $K$, and values $V$ are formed from the flattened CNN feature map with positional encodings added.
- it is `Multi-Head Self-Attention` with standard 8 Heads.
  Each heads have their own $Q$, $K$, $V$.

---
## Transformer Decoder
The decoder is a stack of $N = 6$ decoder layers.
Two things are inputted into the `Decoder`:
- `Object Queries`: Represent each of the $N = 100$ objects
- `Encoder Memory`: Output from encoder of size $HW \times d$

The `Decoder` layer consists of 3 different blocks:
- `Self-Attention Block`
  Queries look at each other so, prevent multiple queries predicting same object.
  $Q' = \text{LayerNorm}(Q + \text{SelfAttn}(Q))$
- `Cross-Attention Block`
  Queries represent objects while Key/Values is the encoder memory (features)
  $\text{Q}'' = \text{LayerNorm}(Q') + \text{CrossAttn}(Q', \text{encoderMemory})$
- `Feed-Forward Network (FFN)`
  Standard `2-Layer MLP` with `RELU` in between.
  $Q_{out} = \text{LayerNorm}(Q'' + \text{FFN}(Q''))$

---
## Prediction Head
Each prediction head is made up of 2 heads
- `Classification Head`
  Linear layer + softmax (classify into each objects)
- `Box Regression Head`
  A 3 layer MLP (linear -> ReLU -> linear -> ReLU -> Linear)

There is a prediction head after each of the 6 `Decoders`.

---
## Loss
The [[Hungarian Algorithm]] + [[Hungarian Loss]] is calculated at the end of each of the $N=6$ prediction head.
Note that all these losses are applied only during the back-propagation.

Each `Decoder` layers get their own respective loss grardients while each `Encoder` and `Prediction Head` get the shared loss gradients of all $N = 6$ losses.

---
## See Also
- [[DETR]]
