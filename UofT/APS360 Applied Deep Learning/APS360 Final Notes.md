# APS360 Final Notes
These are the final notes for my exam.

---
## CNN Dimensions
For [[Convolution Layer|convolution layers]], it is
$$
H_{out} = \frac{H_{in} + 2p - k }{s} +1
$$
$$
W_{out} = \frac{W_{in} + 2p - k }{s} +1
$$

and for [[Pooling Layer|pooling layers]], it is
$$
o = \frac{i - k}{s} + 1
$$

---
## Transformer Dimensions
Attention matrix size is
$$
(\text{batch}, \ \text{heads}, \ \text{seq\_len}, \ \text{seq\_len})
$$

Output embeddings size is
- Per-Head Output Embeddings

$$
(\text{batch}, \ \text{heads}, \ \text{seq\_len}, \ \tilde{d}_{v})
$$
- Concatenate across heads: $\text{heads} \times \tilde{d}_{v} = d_{v}$

$$
(\text{batch}, \ \text{seq\_len}, \ d_{v})
$$

[[Multi-Head Attention|Read More]]

---
## Parameter Count
### Fully Connect Layer
$$
a \times b + b
$$
where
- $a$ is input layer neurons count
- $b$ is output layer neurons count

[[Neural Network|Read More]]

---
## Convolution Layer
$$
(c_{in} \times k \times k + 1) \times c_{out}
$$
where
- $c_{in}$: input depth
- $k$: kernel size
- $1$: bias term
- $c_{out}$: no. of filters(output depth)

[[Convolution Layer|Read More]]

---
## RNN Layer
$$
(d \times h) + (h \times h) + h
$$
where
- $W_{xh} \in \mathbb{R}^{d \times h}$
- $W_{hh} \in \mathbb{R}^{h \times h}$
- $b \in \mathbb{R}^{h}$

[[Recurrent Neural Network (RNN)|Read More]]

---
## LSTM
$$
4\ (hd + h^{2} + h)
$$
where
- $d$ is the input size
- $h$ is the hidden size

[[Long Short-Term Memory (LSTM)|Read More]]

---
## GRU
$$
3\ (hd + h^{2} + h)
$$
where
- $d$ is the input size
- $h$ is the hidden size

[[Gated Recurrent Unit(GRU)|Read More]]

---
## Transformer
$$
\underbrace{4d^2 + 4d}_{\text{Attention (Q,K,V,O)}} + \underbrace{2d \cdot d_{ff} + d_{ff} + d}_{\text{Feedforward}} + \underbrace{4d}_{\text{2 LayerNorms}} = 4d^2 + 2d \cdot d_{ff} + d_{ff} + 9d
$$
where
- $d$ is the dimension of $Q$,$K$,$V$
- $d_{ff}$ is the dimension of the linear layers

[[Transformer|Read More]]

---
## Course Notes
[[APS360 Course Links|APS360 Course Notes]]