# Bidirectional RNN
[[Bidirectional RNN]] process sequence in both directions at once.

![image|400](https://notes-media.kthiha.com/Bidirectional-RNN/9e0f98f3d46aac67dbffecb7ceaeeba9.png)

It takes advantage of the fact that in certain sequential task, a prediction depends on past, present and future.

---
## Bidirectional
- [[Recurrent Neural Network (RNN)|Forward RNN]] reads the sequence from start to end
- [[Recurrent Neural Network (RNN)|Backward RNN]] reads the sequence from end to start

At each time step, the hidden state is obtained by concatentating (or sum/avg) the forward and backward hidden states.

---
## See Also
- [[Recurrent Neural Network (RNN)]]
- [[Bidirectional RNN]]
- [[Deep RNN]]
- [[Sequence to Sequence RNN]]
- [[Gated Recurrent Unit(GRU)]]
- [[Long Short-Term Memory (LSTM)]]
- [[Transformer]]
