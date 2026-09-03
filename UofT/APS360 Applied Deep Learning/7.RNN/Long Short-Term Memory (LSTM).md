# LSTM
[[Long Short-Term Memory (LSTM)|LSTM]] consist of long-term memory(cell state) and short-term memory(context or hidden state).

![image|400](https://notes-media.kthiha.com/Long-Short-Term-Memory-(LSTM)/2dfe97d46d68ce1df938db991974a210.png)

They use three gates to update the memories.

---
## Gating Mechanism
We can approximate [[Skip Connections|skip connections]] to all previous states by learning to weigh previous states differently instead.

![400](https://miro.medium.com/v2/resize:fit:1400/1*ahafyNt0Ph_J6Ed9_2hvdg.png)

[[#Gating Mechanism]] controls how much information flows through.
Suppose $x$ is a vector. Then, we can control how much of $x$ to pass to next step by:
$$
f(x) = x \cdot \sigma(x)
\quad \quad 
f(x) = x \cdot \text{NN}(x)
$$

---
### Forget Gate
[[#Forget Gate]](long-term memory) defines how much of past memory should be forgotten.

![image|200](https://notes-media.kthiha.com/Long-Short-Term-Memory-(LSTM)/980246bc154607fceb7a13a44274d041.png)

---
### Input Gate
[[#Input Gate]](long-term memory) defines how much the current input should contribute to the memory.

![image|400](https://notes-media.kthiha.com/Long-Short-Term-Memory-(LSTM)/6c805ded8e449dcbd2d53eedea7a9153.png)

---
### Updated Long-Term Memory
The [[#Updated Long-Term Memory|updated long-term memory]] is the amount of past that is remembered(decided by [[#Forget Gate|forget gate]]) combined with the memory that was just created (decided by [[#Input Gate|input gate]]).

![image|600](https://notes-media.kthiha.com/Long-Short-Term-Memory-(LSTM)/3f7eac349d2df0a76d9d628fe3d5a26a.png)

---
### Output Gate
[[#Output Gate]](short-term memory) defines how much of the [[#Updated Long-Term Memory|updated long-term memory]] should construct the short-term memory.

![image|400](https://notes-media.kthiha.com/Long-Short-Term-Memory-(LSTM)/8548d4315f394b9c48a7a752dbde4733.png)

---
## Formula
The [[Long Short-Term Memory (LSTM)|forget gate]] is computed as
$$
f_t = \sigma(W_{f} \  x_t + U_f \ h_{t-1} + b_f)
$$
The [[Long Short-Term Memory (LSTM)|input gate]] is computed as
$$
i_t = \sigma(W_i \ x_t + U_i \ h_{t-1} + b_i)
$$
The current [[Long Short-Term Memory (LSTM)|cell state]] is then computed as
$$
\tilde{c}_t = \tanh(W_c \ x_t + U_c \ h_{t-1} + b_c)
$$
The new [[Long Short-Term Memory (LSTM)|cell state]] is computed as
$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
$$
The [[Long Short-Term Memory (LSTM)|output gate]] is computed as
$$
o_t = \sigma(W_o \ x_t + U_o \ h_{t-1} + b_o)
$$
The new [[Long Short-Term Memory (LSTM)|hidden state]] is finally computed as
$$
h_t = o_t \odot \tanh(c_t)
$$
Note that this $h_{t}$ and $c_{t}$ are used in future time-step.

---
## Parameter Count
An [[Long Short-Term Memory (LSTM)|LSTM]] has $3$ gate matrices and $1$ cell matrix.
Each gate can be represented by 
$$
f_{t} = \sigma(W_{x}x_{t} + W_{h}h_{t-1} + b)
$$
This gets us the following parameters:
$$
\underbrace{h\cdot x}_{W_{x}} + 
\underbrace{h\cdot h}_{W_{h}} + 
\underbrace{h}_{b} = h(x+h+1)
$$

Given $3$ gates and $1$ cell state, its parameter count is
$$
4hd + 4h^{2} + 4h
$$
where
- $d$ is the input size
- $h$ is the hidden size

---
## See Also
- [[Recurrent Neural Network (RNN)]]
- [[Bidirectional RNN]]
- [[Deep RNN]]
- [[Sequence to Sequence RNN]]
- [[Gated Recurrent Unit(GRU)]]
- [[Long Short-Term Memory (LSTM)]]