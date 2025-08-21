# State Representation for Hex-Based Strategy HRL

To scale agents from **5×5 → 25×25** hex maps with terrain, units, and fog-of-war, we need **flexible encoders**. Below is a survey of paradigms with **pros/cons** and **reference papers/resources**.

---

## Table of Representation Paradigms

| Paradigm | Concept | Pros | Cons | Reference |
|----------|---------|------|------|-----------|
| **Message-Passing GNNs (GCN, GraphSAGE)** | Nodes exchange messages over edges (tiles, units, adjacency). | Simple, data-efficient, scales to variable map size. | Limited long-range modeling, oversmoothing with depth. | [Kipf & Welling (2016) – GCN](https://arxiv.org/abs/1609.02907) |
| **Attention GNNs (GAT)** | Attention over neighbors with edge features (terrain, direction). | Learns which neighbors matter, handles heterogeneous edges. | Higher compute than GCN; still local. | [Velickovic et al. (2018) – GAT](https://arxiv.org/abs/1710.10903) |
| **Graph Transformers** | Apply Transformer attention to all nodes with positional encodings. | Global reasoning, great for operational/strategic. | O(N²) cost, mitigated with pooling. | [Dwivedi & Bresson (2021)](https://arxiv.org/abs/2106.05234) |
| **Heterogeneous/Relational GNNs (R-GCN)** | Multi-node/edge types (units, supply, rivers, roads). | Rich semantics (logistics, terrain). | More complex design. | [Schlichtkrull et al. (2018)](https://arxiv.org/abs/1703.06103) |
| **Hierarchical/Pooled GNNs (DiffPool, TopK)** | Pool tiles into regions → theaters; multi-scale graph. | Matches tactical→operational→strategic hierarchy. | Pooling adds training complexity. | [Ying et al. (2018) – DiffPool](https://arxiv.org/abs/1806.08804) |
| **Set Transformers / Perceiver IO** | Treat entities (units, cities, bridges) as a set of tokens. | Handles variable # of entities; good for multi-unit interaction. | Need good entity selection to avoid overload. | [Lee et al. (2019) – Set Transformer](https://arxiv.org/abs/1810.00825); [Jaegle et al. (2021) – Perceiver IO](https://arxiv.org/abs/2107.14795) |
| **Hybrid GNN + Set Attention** | Local GNN encodes; Set Transformer aggregates entities globally. | Best of local + global. | More compute, careful fusion needed. | [Kossen et al. (2021)](https://arxiv.org/abs/2106.04566) |
| **CNN on Hex Grids** | Project hex → skewed 2D grid; use CNN with hex kernels/masks. | Efficient, well-optimized libs, good local patterns. | Less flexible for irregular maps, poor global context. | [Cohen et al. (2017) – Group Equivariant CNNs](https://arxiv.org/abs/1602.07576) |
| **Region Graphs (super-tiles)** | Cluster tiles into regions (basins, road nets). | Smaller N, natural for operational/strategic. | Requires clustering algorithm, dynamic updates. | [Bojchevski et al. (2020)](https://arxiv.org/abs/2006.05205) |
| **World Models (PlaNet, Dreamer)** | Learn latent dynamics for planning in imagination. | Long-horizon planning without environment calls. | Harder to train, extra model. | [Hafner et al. (2019) – PlaNet](https://arxiv.org/abs/1811.04551); [Hafner et al. (2020) – Dreamer](https://arxiv.org/abs/1912.01603) |
| **Graph + Memory (GRU/LSTM, Transformer)** | Keep memory of last-seen observations under fog. | Handles partial observability. | Training stability harder. | [GTrXL – Parisotto et al. (2020)](https://arxiv.org/abs/1910.06764) |

---

## Hybrid Encoder Design (Text Diagram)

```
Input: Hex map (5x5 → 25x25), terrain, units, supply, fog

Step 1: Local Tactical Encoding
  - Build local subgraph around each controlled unit (radius-r hex neighborhood).
  - Node features: terrain type, elevation, supply, ownership, unit stats, axial coords.
  - Edge features: adjacency (direction 0–5), river/road flags, movement cost, LOS.
  - Model: 2–3 layer GAT → produces per-unit embeddings.
  - Head: Option-Critic policies (tactical skills).

Step 2: Region Graph Construction
  - Cluster tiles into regions (e.g., connected road nets, river basins, operational sectors).
  - Aggregate tactical embeddings into region tokens (mean/attention pooling).
  - Build region-level graph with inter-region edges (supply lines, adjacency).
  - Model: Graph Transformer with spectral + centroid positional encodings.
  - Head: HIRO goal generator + feasibility scoring (operational level).

Step 3: Strategic Abstraction
  - Pool regions → strategic theaters (high-level clusters).
  - Transformer over theater tokens for global reasoning.
  - Head: MAXQ root subtask selector (strategic objectives).

Step 4: Memory & Fog Handling
  - Recurrent GRU/Transformer memory per region token to maintain beliefs.
  - Update embeddings with last-seen info when fogged.

Output:
  - Tactical: per-unit option/action logits.
  - Operational: subgoals (coordinates/latents).
  - Strategic: subtask selection (capture, defend, resupply).
```

---

## TL;DR
- Use **GNNs (GCN/GAT)** for **tactical local reasoning**.  
- Use **region graphs + Graph Transformers** for **operational/strategic global planning**.  
- Use **Set Transformers or hybrids** when entity interactions dominate (multi-unit battles).  
- Add **memory modules** for fog-of-war.  
- This multi-level hybrid encoder aligns naturally with the HRL hierarchy.  

---
