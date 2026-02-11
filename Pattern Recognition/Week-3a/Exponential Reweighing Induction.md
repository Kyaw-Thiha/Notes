# Exponential Reweighing Induction
> The distribution $D^{(t+1)}$ used in round $t+1$ can be expressed as

$$
D_{i}^{(t+1)}
= \frac{e^{-y_{i} f_{t}(x_{i})}}
{\sum^m_{j=1} e^{-y_{j} f_{t}(x_{j})}}
$$

---
## Proof
Define the `inductive hypothesis` $P(t)$ as
$$
P(t):
D_{i}^{(t)} = \frac{e^{-y_{i} f_{t-1} (x_{i})}}{Z_{t-1}}
$$
where
- $\sum^m_{j=1} e^{-y_{j} f_{t-1}(x_{j})}$ is the `normalization constant`.
- $\sum^{t-1}_{p=1} w_{p} h_{p}(x_{i})$ is the `cumulative weighted score`.

We will be proving by `weak induction`.

---
### Base Case

At $t=1$, 
- $f_{0}(x) = 0$ since there is no iterations yet
- $D_{i}^{(1)} = \frac{1}{m}$ since the distribution is uniformly initialized

Suppose $P(1)$.
Hence,
$$
D_{i}^{(t)}
= \frac{e^{-y_{i} \ \cdot \ 0}}
{\sum^m_{j=1} e^{-y_{j} \ \cdot \ 0}}
= \frac{e^{0}} {\sum^m_{j=1} e^{0}}
= \frac{1} {m}
$$

Thus, $P(1)$ holds.

---
### Induction Step
Given that $P(t)$ holds, prove that $P(t+1)$ holds.
**Induction Hypothesis**: $P(t)$ holds.

Then,
$$
D_{i}^{(t+1)}
= \frac{D^{(t)}_{i} \cdot e^{-y_{i} w_{t} h_{t}(x_{i})}}{Z_{t}}
$$

By induction hypothesis, we get
$$
\begin{align}
D_{i}^{(t+1)}
&= \frac{D^{(t)}_{i} \cdot e^{-y_{i} w_{t} h_{t}(x_{i})}}{Z_{t}} \\[6pt]

&= \frac{\frac{e^{-y_{i} f_{t-1} (x_{i})}}{Z_{t-1}} \cdot e^{-y_{i} w_{t} h_{t}(x_{i})}}{Z_{t}} \\[6pt]

&= \frac{e^{-y_{i} f_{t-1} (x_{i})}  
\cdot e^{-y_{i} w_{t} h_{t}(x_{i})}} 
{Z_{t-1} \cdot Z_{t}} \\[6pt]

&= \frac{e^{-y_{i} f_{t-1} (x_{i}) - y_{i} w_{t} h_{t}(x_{i})} }
{Z_{t-1} \cdot Z_{t}} \\[6pt]

&= \frac{e^{-y_{i} ( f_{t-1} (x_{i}) + w_{t} h_{t}(x_{i}))} }
{Z_{t-1} \cdot Z_{t}} \\[6pt]

&= \frac{e^{-y_{i} \ f_{t}(x_{i}) } }
{Z_{t-1} \cdot Z_{t}} & (1) \\[6pt]

&= \frac{e^{-y_{i} \ f_{t}(x_{i}) } }
{\sum^m_{j=1} e^{-y_{j} f_{t}(x_{j})}} & (2) \\[6pt]
\end{align}
$$
Thus, $P(t+1)$ holds.

$(1)$: By definition of `cumulative score` at iteration $t$,
$$
f_{t}(x_{i}) = f_{t-1}(x_{i}) + w_{t} h_{t}(x_{i})
$$

$(2)$: By defining the `cumulative normalization constant`,
$$
Z_{t}' := Z_{t-1} \cdot Z_{t}
= \sum^m_{j=1} e^{-y_{j} \ f_{j}(x_{j})}
$$

---
## Conclusion
By `induction`, $P(t)$ holds for $\forall t\geq 1$.
Therefore, any algorithm using `margin-based exponential reweighting` with `cumulative score` $f_{t}$ will produce a distribution of the form:

$$
D_{i}^{(t+1)}
= \frac{e^{-y_{i} f_{t}(x_{i})}}
{\sum^m_{j=1} e^{-y_{j} f_{t}(x_{j})}}
$$

---
## See Also
- [[AdaBoost Exponential Convergence]]
- [[Maths behind AdaBoost]]
