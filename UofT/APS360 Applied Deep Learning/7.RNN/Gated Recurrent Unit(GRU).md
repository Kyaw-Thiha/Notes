# GRU
[[Gated Recurrent Unit(GRU)|GRU]] works similar to [[Long Short-Term Memory (LSTM)|LSTMs]] but with less gates.

![image|400](https://notes-media.kthiha.com/Gated-Recurrent-Unit(GRU)/2aa4e1e0ddf9603c2fe8f97b95f96675.png)

---
## Comparism to LSTM

![350](https://media.geeksforgeeks.org/wp-content/uploads/20251215093334187826/gru_vs_lstm.webp)

Compared to [[Long Short-Term Memory (LSTM)|LSTMs]],
- they combine [[Long Short-Term Memory (LSTM)|forget]] and [[Long Short-Term Memory (LSTM)|input gates]] into an [[Gated Recurrent Unit(GRU)|update gate]]
- merges cell state and hidden state
- thus are more efficient than [[Long Short-Term Memory (LSTM)|LSTMs]] for similar performance

---
## Formula
The [[Gated Recurrent Unit(GRU)|update gate]] is computed as
$$
z_t = \sigma(W_z \ x_t + U_z \ h_{t-1} + b_z)
$$

The [[Gated Recurrent Unit(GRU)|reset gate]] is computed as
$$
r_t = \sigma(W_r \ x_t + U_r \ h_{t-1} + b_r)
$$

The [[Gated Recurrent Unit(GRU)|candidate hidden state]] is then computed as
$$
\tilde{h}_t = \tanh(W_h \ x_t + U_h \ (r_t \odot h_{t-1}) + b_h)
$$

The new [[Gated Recurrent Unit(GRU)|hidden state]] is finally computed as
$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$
Note that this $h_{t}$ is used in the future time-step.

---
## Parameter Count
An [[Gated Recurrent Unit(GRU)|GRU]] has $2$ gate matrices and $1$ hidden state matrix.
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

Given $2$ gates and $1$ hidden state, its parameter count is
$$
3hd + 3h^{2} + 3h
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
- [[Transformer]]
