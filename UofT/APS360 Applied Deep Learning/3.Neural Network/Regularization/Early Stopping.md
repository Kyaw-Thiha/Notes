# Early Stopping
[[Early stopping]] is a form of [[regularization]] by stopping the training before the model overfits.

![|300](https://miro.medium.com/v2/resize:fit:4800/format:webp/1*-37i5H3lie_x5299utHX4A.png)

Monitor the model performance on the [[Cross Validation|validation set]] during training and stop the training when performance starts to degrade.

---
## Early Stopping with Patience
- In each training iteration, observe the [[Cross Validation|validation]] [[Loss Function|loss]].
- As soon as [[Loss Function|validation loss]] starts to increase, start a counter.
- If the [[Loss Function|validation loss]] decrease, reset the counter.
- Otherwise, wait for a fixed iterations(`patience`) and then stop the training.

![image|300](https://notes-media.kthiha.com/Early-Stopping/7b331e29e98652c02c12b8801da67bf0.png)

---
## See Also
- [Good Article](https://cyborgcodes.medium.com/what-is-early-stopping-in-deep-learning-eeb1e710a3cf)
- [[Regularization]]
- [[Weight Decay]]
- [[Early Stopping]]
- [[Dropout]]
