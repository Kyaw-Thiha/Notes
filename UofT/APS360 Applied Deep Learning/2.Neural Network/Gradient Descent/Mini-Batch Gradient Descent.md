# Mini-Batch Gradient Descent
Instead of one sample per time as in [[Stochastic Gradient Descent(SGD)|SGD]] or entire batch as in [[Gradient Descent|batch gradient descent]], we can apply mini-batches:
- Use the [[Neural Network|network]] to make predictions for $n$ samples
- Compute average loss for $n$ samples
- Take a step to optimize the average loss for $n$ samples

---
## Terminology
- `Batch Size`: No. of training examples per optimization step.
- `Iteration`: Parameters are updated once per iteration.
- `Epoch`: No. of times all train data is used.

Eg: Suppose there are $1000$ samples in training dataset. If we set batch size to $10$, then $1$ epoch will contain $100$ iterations.

---
## Ineffective Batch Size
If the batch size is too small,
- We optimize different [[Loss Function|function loss]] at each iteration.
- Noisy, just like [[Stochastic Gradient Descent(SGD)|SGD]].

If the batch size is too large,
- Expensive on memory and compute.
- Average loss might not change very much (lower variance).
- True gradient is not always the best gradient for optimization.

---
## Optimizers
- [[Gradient Descent]]
- [[Gradient Descent Detail]]
- [[Stochastic Gradient Descent(SGD)]]
- [[SGD with Momentum]]
- [[Nesterov Accelerated Gradient(NAG)]]
- [[Adagrad]]
- [[Adadelta]]
- [[RMS Prop]]
- [[Adaptive Moment Estimation(Adam)]]
- [[AdamW]]

---
## See Also
- [Good Blog by Sebastian Ruder](https://www.ruder.io/optimizing-gradient-descent/)
- [Blog covering SGD to latest methods](https://k4i.top/posts/optimizers-adamw/#from-update-to-sgd)
- [Paper with mathematical proofs of different optimizers](https://arxiv.org/pdf/2501.14458)