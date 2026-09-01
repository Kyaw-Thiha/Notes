# Multi-Head Attention
We could use multiple of the [[Self-Attention|self-attention blocks]] in parallel.

![300](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSHY48TjVeZbySyGQAmqWoegmDM8CKJ509cdL7Q3dIs-IQFTOaKylb7VSXE&s=10) 

To improve the performance,
- Divide the representation space to $h$ sub-spaces
- Run parallel [[Neural Network|linear layers]] and [[Self-Attention|attention]]
- Concatenate them back to form the original space

$$
\text{Multi-Head}(Q,K,V)
= \text{Concat}(\text{head}_{1}, \ \dots, \ \text{head}_{h}) \ W^{O}
$$
where $\text{head}_{i} = \text{Attention}(QW^{Q}_{i}, \ KW^{K}_{i}, \ VW^{V}_{i})$.

---
### Attention Matrix
The [[Self-Attention|attention tensor]] is the [[Attention Mechanism|attention weight matrix]]. This is the output of 
$$
\text{softmax}\left( \frac{QK^{T}}{\sqrt{ d_{k} }} \right)
$$
computed separately per [[Multi-Head Attention|head]].

### Size
$$
(\text{batch}, \ \text{heads}, \ \text{seq\_len}, \ \text{seq\_len})
$$

> **Important**
> Note that the dimension of query, key, and values have to be divided by the no. of heads.

---
### Queries & Keys
$Q$ and $K$ each have shapes of:
$$
(\text{batch}, \ \text{heads}, \ \text{seq\_len}, \ d_{k})
$$
Inside $\text{mathmul } QK^{T}$, transposing $K$'s last two dims give shape of $(\text{batch}, \ \text{heads}, \ d_{k}, \ \text{seq\_len})$, so the multiply is
$$
(\text{seq\_len} \times d_{k}) \cdot
(d_{k} \times \text{seq\_len})
\to (\text{seq\_len} \times \text{seq\_len})
$$
where each entry $(i, \ j)$ of the output is the dot product of query vector $i$ and key vector $j$:
$$
(QK^{T})_{ij}
= \sum^{d_{k}}_{m=1} Q_{im} K_{jm}
$$
which is a sum of $d_{k}$ products.

---
#### Scaling Factor
Each dot product $(QK^{T})_{ij}$ has variance that grows proportionally to $d_{k}$. Dividing by $\sqrt{ d_{k} }$ keeps variance of logits roughly constant $(\approx 1)$ regardless of how large $d_{k}$ is.

---
### Output Tensor of Embeddings
- Attention Matrix: $\text{softmax}\left( \frac{QK^{T}}{\sqrt{ d_{k} }} \right)$. 

$$
(\text{batch}, \ \text{heads}, \ \text{seq\_len}, \ \text{seq\_len})
$$

- Update $d_{v}$ (and $d_{k}, d_{q}$) by dividing by $\text{heads}$.

$$
\tilde{d}_{v} = \frac{d_{v}}{\text{heads}}
$$

- Multiply by values: $A \cdot V$

$$
(\text{seq\_len} \times \text{seq\_len})
\cdot
(\text{seq\_len} \times d_{v})
= \text{seq\_len} \times \tilde{d}_{v}
$$


#### Sizes
- Per-Head Output Embeddings

$$
(\text{batch}, \ \text{heads}, \ \text{seq\_len}, \ \tilde{d}_{v})
$$
- Concatenate across heads: $\text{heads} \times \tilde{d}_{v} = d_{v}$

$$
(\text{batch}, \ \text{seq\_len}, \ d_{v})
$$


---
## See Also
- [[Transformer]]
- [[Attention Mechanism]]
- [[Attention Mask]]
- [[Multi-Head Attention]]
- [[Understanding Projections in Self-Attention (Q, K, V)]]
- [Vaswani et al., "Attention is All You Need" (2017)](https://arxiv.org/pdf/1706.03762)
