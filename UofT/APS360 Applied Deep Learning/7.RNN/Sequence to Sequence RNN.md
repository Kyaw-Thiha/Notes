# Sequence to Sequence RNN
[[Sequence to Sequence RNN]] learns to predict new sequences based on training data of sequences.

---
## During Training
### BOS/EOS
We use dedicated [[#BOS/EOS|Beginning of Sequence]](`<BOS>`) and [[#BOS/EOS|End of Sequence]](`<EOS>`).

---
### Ground Truth and Loss
The [[Recurrent Neural Network (RNN)|RNN]] is trained to generate one particular sequence in the training set.

![image|500](https://notes-media.kthiha.com/Sequence-to-Sequence-RNN/21d8d4c66614a29fb5f2b2d1419d57c0.png)

In each step, we compute the [[Loss Function|loss]] by comparing ground-truth and predicted tokens.

---
### Teacher Forcing
In order to make training more efficient , we force the [[Recurrent Neural Network (RNN)|RNN]] to stay close to the ground-truth sequence.

We do this by passing the ground-truth label as the next input instead of current prediction.

---
## During Inference
Unlike in classification problem, always selecting token with highest probability won't work well:
- We want diversity not deterministic behaviour
- Greedy approach results in lots of grammatical error

To address this we sample from predicted distributions.

---
### Greedy Search
[[#Greedy Search]] selects the token with highest probability as the generated token.
$$
\max p(t_{1}, t_{2}, \dots, t_{n}) 
= \max \left( \ p(t_{1}) \times p(t_{2}) \times \dots 
\times p(t_{n}) \ \right)
$$

---
### Beam Search
[[#Beam Search]] looks or a sequence of tokens with the highest probability within a window.

![300](https://cdn.amazon.science/5c/dc/3969f3e44740956595c181578589/beam-search.gif)

$$
\max p(t_{1}, t_{2}, \dots, t_{n}) 
= \max \left( \ p(t_{1}) \times p(t_{2}\mid t_{1})
\times \dots \times p(t_{n}\mid t_{n-1}, 
\dots t_{2, t_{1}}) \ \right)
$$

---
### Softmax Temperature Scaling
[[#Softmax Temperature Scaling]] helps with the problem of over-confidence in [[Neural Network|neural network]] by scaling the input logits to the softmax with a temperature.

![300](https://miro.medium.com/v2/resize:fit:800/1*p1iKxUJcXDlSEZCpMCwNgg.gif)

$$
\text{softmax}(z_{i})
= \frac{e^{z_{i}/\tau}}{\sum_{i} e^{z_{i}/\tau}}
$$

- **Low Temperature**: larger logits, more confidence
  Higher quality samples, less variety
- **High Temperature**: smaller logits, less confident
  Lower quality samples, more variety

---
## See Also
- [[Recurrent Neural Network (RNN)]]
- [[Bidirectional RNN]]
- [[Deep RNN]]
- [[Sequence to Sequence RNN]]
- [[Gated Recurrent Unit(GRU)]]
- [[Long Short-Term Memory (LSTM)]]
- [[Transformer]]