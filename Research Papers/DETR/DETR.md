# DETR (Detection Transformer)
#research #cv/object-detection/models/detr

Link: https://arxiv.org/abs/2005.12872

`DETR` is an end-to-end model that leverage transformers to remove hand-crafted preprocessing methods.

![Architecture](https://viso.ai/wp-content/uploads/2024/02/DETR-Architecture.jpg)


Most prior models use [[Region Proposals]], [[Window Centers]] and [[Anchors]] to predict objects and learn from them.

All of these methods require additional handcrafted preprocessing layer like [[Non-Maximum Suppression|NMS]] which add computational cost during inference.

To solve this, there has been prior models that also use [[Bipartite Matching]] but they also require additional handcrafted components.
Recurrent Neural Networks (RNNs) have also been considered, but they are autoregressive, and hasn't been tested on large datasets.

To solve this issue, `DETR` apply a specific [[Bipartite Matching]] algorithm for better learning, and propose a novel architecture built with a ResNet backbone, and transformer stacked on top of it.

## Bipartite Matching
The model has a fixed number of 100 predictions per image.
These model undergoes [[Bipartite Matching]] whereby a $100 \times 100$ cost matrix is computed through [[Matching Cost]], and used for matching through [[Hungarian Algorithm]].

These the loss of these matched pairs are computed through [[Hungarian Loss]], which is used to updates the neurons.

## Architecture
![Architecture](https://viso.ai/wp-content/uploads/2024/02/DETR-Architecture.jpg)

Three main components
- CNN backbone
- Encoder-Decoder Transformer
- FFN Prediction Heads

[[DETR Architecture|Read More]]

---
## See Also
- [[DETR Architecture]]
- [[Bipartite Matching]]
- [[Matching Cost]]
- [[Hungarian Algorithm]]
- [[Hungarian Loss]]
- [DETR Explanation](https://viso.ai/deep-learning/detr-end-to-end-object-detection-with-transformers/)
- [Research Paper](https://arxiv.org/abs/2005.12872)