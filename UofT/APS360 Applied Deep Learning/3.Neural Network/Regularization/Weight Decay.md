# Weight Decay
[[Weight decay]] is a form of [[regularization]] by preventing  weights from growing too much and hence lowering the variance.

Add a [[Norm|2-norm]] to the [[loss function]] as
$$
L(W; \ y,t)
= L(W; \ y,t) + \frac{\alpha}{2} ||W||^{2}_{2}
$$
Then, the weights would be updated as
$$
\begin{align}
&\frac{\delta E}{\delta W}  
= \frac{\delta E}{\delta W} + \alpha W \\[6pt]

&W_{t+1} = W_{t} - \gamma\left( \alpha W_{t}  
+ \frac{\delta E}{\delta W} \right)
\end{align}
$$
Note that the decay is multiplicative and proportional to $W$.

---
## See Also
- [[Regularization]]
- [[Regularized Loss Minimization (RLM)]]
- [[Weight Decay]]
- [[Early Stopping]]
- [[Dropout]]
