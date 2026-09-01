# ViT
[[ViT(Vision Transformer)|Vision transformers]] adapt the [[Transformer|transformer architecture]] to computer vision tasks.

![400](https://viso.ai/wp-content/uploads/2021/09/vision-transformer-vit.png)

---
## How it works
- **Patchify**: Image is split into patches.
- **Linear Embedding**: Each patch is flattened into a vector and projected into an embedding space.
- **Position embeddings**: learned position embeddings are added to each patch embedding.
- **[CLS] token**: Special learnable token is prepended to the image sequence.
- **Transformer Encoder**.
- **Classification Head**.

---
## Comparism to CNN
Compared to [[Convolutional Neural Network (CNN)|CNN]], they have
- higher modelling capability
- lower inductive bias
- global receptive field

---
## See Also 
- [Paper Introducing ViTs](https://arxiv.org/abs/2010.11929)
- [[Transformer]]
- [[Multi-Head Attention]]
- [[Self-Attention]]
- [[Convolutional Neural Network (CNN)]]