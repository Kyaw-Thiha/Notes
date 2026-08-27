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
