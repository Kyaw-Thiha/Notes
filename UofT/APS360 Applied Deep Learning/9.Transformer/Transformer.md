# Transformer
[[Transformer]] are [[Neural Network|deep learning models]] that are based on the [[Self-Attention|self-attention]] mechanism.

![300](https://media.geeksforgeeks.org/wp-content/uploads/20251210153206327851/transformers.webp)

---
### Attention
It retrieves value $v_{i}$ for query $q$, based on key $k_{i}$.
$$
\text{attention}(q,k,v)
= \sum_{i} \text{similarity}(q, k_{i}) 
\times v_{i}
$$

---
#### Linear Projection
Suppose $X$ is an input sequence consisting of $n$ tokens where each token $t \in \mathbb{R}^{i}$. To compute the queries, keys, and values from the input sequence $X$, we use three linear layers:
$$
\begin{align}
Q = \textbf{XW}_{Q}  \\[6pt]
K = \textbf{XW}_{K}  \\[6pt]
V = \textbf{XW}_{V}  \\[6pt]
\end{align}
$$
where 
- $\mathbf{X} \in \mathbb{R}^{n \times i}$ is the input sequence.
- $\mathbf{W}_{Q} \in \mathbb{R}^{i \times k}, \ \mathbf{W}_{K}\in \mathbb{R}^{i \times k}, \ \mathbf{W}_{V} \in \mathbb{R}^{i \times v}$ are weight matrices.
- $\mathbf{Q} \in \mathbb{R}^{n \times k}$ is the query.
- $\mathbf{K} \in \mathbb{R}^{n \times k}$ is the key.
- $\mathbf{V} \in \mathbb{R}^{n \times v}$ is the value.

and
- $n$ is the no. of tokens.
- $i$ is the dimension of each token $t \in \mathbb{R}^{i}$.

---
#### Self-Attention Computation
We compute the pairwise attention score between all tokens within the input sequence $X \in \mathbb{R}^{n \times n}$.

![300](https://miro.medium.com/v2/resize:fit:1400/1*MgMP9-ewpcZsgSvPgcHgxg.png)

It is defined as
$$
\text{Attention}(Q,K,V)
= \text{softmax}( \frac{Q K^{T}}{\sqrt{ d_{k} }} ) \ V
$$
Note that hidden dimension $d_{k}$ of $Q,\ K$ is used to keep the scores from blowing up.

Also $\text{Attention}(Q,K,V) \in \mathbb{R}^{n \times v}$.

---
### Multi-Head Attention
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
### Transformer Encoder
Each [[#Transformer Encoder|encoder]] layer consists of
- A [[Self-Attention|multi-head self-attention]] sub-layer.
- A [[Neural Network|fully connected]] sub-layer.
- A residual connection around each of the $2$ sub-layers followed by [[Layer Normalization|layer normalization]].

![120](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTndmdiFMkNSgvRlziU_uduJRhCi2g_FmI1PY0jjTAJkdZPbh4tZY3szZg&s=10)

It can be defined as 
- $\text{MultiHead(Q,K,V)} = \text{concat}(\text{head}_{1}, \ \dots, \ \text{head}_{h}) \ W^{O}$
- $\text{FFN}(x) = \max(0 , xW_{1} + b_{1}) \ W_{2} + b_{2}$
- $\text{LayerNorm}(x + \text{Sublayer}(x))$

---
### Positional Encoding
[[#Positional Encoding]] allows the model to easily learn to attend by relative positions by making use of order.

![300](https://machinelearningmastery.com/wp-content/uploads/2022/01/PE3.png)

It uses the formula of
$$
\begin{align}
\text{PosEncoding}_{(pos, 2i)}
&= \sin\left( \frac{pos}{10000^{2i/d_{model}}} \right) \\[6pt]

\text{PosEncoding}_{(pos, 2i+1)}
&= \cos\left( \frac{pos}{10000^{2i/d_{model}}} \right) \\[6pt]
\end{align}
$$

---
### Comparism to RNN
For the [[Recurrent Neural Network (RNN)]],
- Struggle with long range dependencies.
- [[Vanishing and Exploding Gradients|Gradients vanishing and exploding]].
- Require large no. of training steps.
- Recurrence prevents parallel computation.

For the [[Transformer]], 
- Facilitate long range dependencies.
- Less likely to have [[Vanishing and Exploding Gradients|Gradients vanishing and exploding]] problem.
- Require fewer training steps.
- No recurrence, facilitates parallel computation.

---
## See Also
- [A Good Blog](https://www.datacamp.com/tutorial/how-transformers-work)
- [Code Examples](https://nlp.seas.harvard.edu/annotated-transformer/)
- [Code Examples (Old)](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [[Transformer]]
- [[Transformer for Language Modelling]]
- [[Self-Attention]]
- [[Attention Mechanism]]
- [[Recurrent Neural Network (RNN)]]
- [[Word2Vec]]
- [[GLoVe]]

