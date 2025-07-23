#ml #llm/transformers/encoders/positional-encoding 
# 🌀 Mathematical Explanation of Positional Encoding

Since the Transformer has no recurrence or convolution, it needs an explicit way to encode the **order of tokens** — this is done via **positional encoding**. The original Transformer paper proposed **sinusoidal positional encodings**, which are deterministic and fixed (not learned).

Given a sequence position $pos$ and a dimension index $i$, the encoding for dimension $i$ is defined as:

$$
\text{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{\frac{2i}{d_{\text{model}}}}}\right), \quad
\text{PE}_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{\frac{2i}{d_{\text{model}}}}}\right)
$$

This alternating sine/cosine formulation ensures that nearby positions have similar encodings and that the position information generalizes to sequences longer than those seen during training. Each token’s embedding is added with its corresponding positional encoding vector:

$$
\text{Input}_\text{final} = \text{Embedding} + \text{PositionalEncoding}
$$

Importantly, these encodings allow the model to infer **relative and absolute position information** through the dot-product attention mechanism, even though the model itself is position-agnostic.

## See Also
- [[Encoder]]
- [[Positional Encoding]]