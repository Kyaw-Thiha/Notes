# AdaBoost Exponential Convergence
> `Proof`: Training Error of [[AdaBoost]] drops exponentially fast.

---
## Theorem
Let $S$ be a `training set`.
Assume that at each iteration of [[AdaBoost]], the weak learner returns a hypothesis for which
$$
\epsilon_{t} \leq \frac{1}{2} - \gamma
$$
where
- $\epsilon_{t} = \sum_{i:h_{t}(x_{i}) \neq y_{i}} D_{t}(i)$ is the `weighted error` of weak learner
- $\leq \frac{1}{2} - \gamma$  means that error is less than $\frac{1}{2}$.
  Hence, it is better than `random guessing`

---
Then, the training error of the output hypothesis of [[AdaBoost]] is at most
$$
\begin{align}
L_{S}(h_{S})  
= \frac{1}{m} \sum^m_{i=1}
\mathbb{1}(h_{S}(x_{i}) \neq y_{i})
&\leq \exp(-2 \gamma^2 T) \\[6pt]
\end{align}
$$
where
- $L_{S}(h_{S})$ is the [[Empirical Risk]](training error) of the final [[AdaBoost]] hypothesis
- $h_{S} = \text{sign}\left( \sum^T_{t=1} \alpha_{t} h_{t}(x) \right)$ is the `final hypothesis output` by [[AdaBoost]].
  Its a `weighted majority vote` of the [[Weak Learner|weak learners]].
- $m$ is the no. of training examples in set $S$
- $T$ is the number of boosting iterations

> The bound $\exp(-2 \gamma^2 T)$ show that training error decreases exponentially with respect to `number of rounds` $T$ and the `square of the edge` $\gamma^2$.

---
## Proof
For each $t$, denote 
$$f_{t} = \sum_{p\leq t} w_{p}h_{p}$$
where
- $f_{t}$ is the `cumulative weighted hypothesis` after $t$ rounds
- $h_{p}$ is the `weak hypothesis classifier` at round $p$
- $w_{p}$ is the `weight` of hypothesis $h_{p}$

Therefore, the output of [[AdaBoost]] is $f_{T}$.

In addition, denote
$$
Z_{t} 
= \frac{1}{m} \sum^m_{i=1} e^{-y_{i} f_{t}(x_{i})}
$$
Note that for any hypothesis, $\mathbb{1}(h(x) \neq y) \leq e^{-y \ h(x)}$ ([[AdaBoost Error Bound|Proof here]]) 
$\therefore$ $L_{S}(f_{t}) \leq Z_{T}$

So, all we have to show is 
$$
Z_{T} \leq \exp(-2 \gamma^2 \ T)
$$

---
### Proof: Bounding
To upper bound $Z_{T}$, we rewrite it as a `telescoping sum`:
$$
Z_{T}
= \frac{Z_{T}}{Z_{0}}
= \frac{Z_{T}}{Z_{T-1}}
\ \frac{Z_{T-1}}{Z_{T-2}}
\ \dots \ \frac{Z_{2}}{Z_{1}} \frac{Z_{1}}{Z_{0}}
= \prod^{T-1}_{t=0} \frac{Z_{t+1}}{Z_{t}}
$$

since $f_{0} \triangleq 0$ and $Z_{0} = 1$.

Therefore, we just need to show that for every round $t$,
$$
\frac{Z_{t+1}}{Z_{t}}
\leq e^{-2 \gamma^2}
$$

---
### Using Inductive Argument
First, form the inductive argument that for all $t$ and $i$:
$$
D_{i}^{(t+1)}
= \frac{e^{-y_{i} f_{t}(x_{i})}}
{\sum^m_{j=1} e^{-y_{j} f_{t}(x_{j})}}
$$
Proof for the argument can be found [[Exponential Reweighing Induction|here]].

Hence, 
$$
\begin{align}
\frac{Z_{t+1}}{Z_{t}}

&= \frac{\sum^m_{i=1} e^{-y_{i} \ f_{t+1}(x_{i})}} 
{\sum^m_{j=1} e^{-y_{j} \ f_{t}(x_{j})}} \\[6pt]

&= \frac{\sum^m_{i=1} e^{-y_{i} \ f_{t}(x_{i})} \ e^{-y_{i} w_{t+1} h_{t+1}(x_{i})}} 
{\sum^m_{j=1} e^{y_{j} f_{t}(x_{j})}}  
& \text{by (1)}\\[6pt]

&= \sum^m_{i=1} D_{i}^{(t+1)}  
e^{-y_{i} \ w_{t+1} h_{t+1}(x_{i})}  \\[6pt]

&= \left(e^{-w_{t+1}} \sum_{i:y_{i} h_{t+1}(x_{i}) = 1}
D_{i}^{(t+1)} \right)  
+ \left(e^{w_{t+1}}
\sum_{i:y_{i} \ h_{t+1}(x_{i}) = -1}
D_{i}^{(t+1)} \right) & \text{by (2)} \\[6pt]

&= e^{-w_{t}+1}(1 - \epsilon_{t+1})
+ e^{w_{t+1}} \epsilon_{t+1}  
& \text{by } (3) \\[6pt]

&= \frac{1}{\sqrt{ \frac{1}{\epsilon_{t+1}} - 1 }}
(1 - \epsilon_{t+1})
+ \sqrt{ \frac{1}{\epsilon_{t+1} - 1} } (\epsilon_{t+1})
& \text{by } (4)
\end{align}
$$
where
- $(1):$ By recursive definition of [[AdaBoost]] combined hypothesis.
$$
f_{t+1}(x)
= f_{t}(x) + w_{t+1} h_{t+1}(x)
$$
- $(2)$: Split the sum $\sum^m_{i=1} D_{i}^{(t+1)} e^{-y_{i} \ w_{t+1} h_{t+1}(x_{i}) }$ based on $2$ cases.
	- **Case-1**: Correct Prediction $\implies y_{i}h_{t+1}(x_{i}) = +1$ 
$$
e^{-y_{i} \ w_{t+1}h_{t+1}(x_{i})}
= e^{-w_{t+1}\cdot(+1)}
= e^{-w_{t+1}}
$$
	- **Case-2**: Incorrect Prediction $\implies y_{i} h_{t+1}(x_{i}) = -1$ 
$$
e^{-y_{i} \ w_{t+1}h_{t+1}(x_{i})}
= e^{-w_{t+1}\cdot(-1)}
= e^{+w_{t+1}}
$$
	- Combining them together

$$
\begin{align}
&\sum^m_{i=1} D_{i}^{(t+1)} \  
e^{-y_{i} \ w_{t+1}h_{t+1}(x_{i})} \\[6pt]

&= \sum_{i:y_{i}:h_{t+1}(x_{i}) = 1}
D_{i}^{(t+1)} e^{-w_{t+1}}
+ \sum_{i:y_{i}:h_{t+1}(x_{i}) = -1}
D_{i}^{(t+1)} e^{+w_{t+1}} \\[6pt]

&= \left(e^{-w_{t+1}} \sum_{i:y_{i} h_{t+1}(x_{i}) = 1}
D_{i}^{(t+1)} \right)  
+ \left(e^{w_{t+1}}
\sum_{i:y_{i} \ h_{t+1}(x_{i}) = -1}
D_{i}^{(t+1)} \right) \\[6pt]
\end{align}
$$

- $(3)$: Let $\epsilon_{t+1}$ be the `sum of weights on incorrectly classified examples`.
$$
\epsilon_{t+1}
= \sum_{i: y_{i} h_{t+1}(x_{i}) = -1}
D_{i}^{(t+1)}
$$

- $(4a)$: In [[AdaBoost]], the weight $w_{t+1}$ for a [[Weak Learner]] is chosen as

$$
w_{t+1} 
= \frac{1}{2} \ln\left( \frac{1 - \epsilon_{t+1}}
{\epsilon_{t+1}} \right)
$$
$$
\begin{align}
&\text{Then, } \\[6pt]
&w_{t+1} = \frac{1}{2} \ln \left(  \frac{1 - \epsilon_{t+1}}{\epsilon_{t+1}} \right) \\[6pt]

&e^{w_{t+1}} = e^{\frac{1}{2} \ln \left(  \frac{1 - \epsilon_{t+1}}{\epsilon_{t+1}} \right)} \\[6pt]

&e^{w_{t+1}} = \sqrt{ \frac{1 - \epsilon_{t+1}} 
{\epsilon_{t+1}} } \\[6pt]

&e^{w_{t+1}} = \sqrt{ \frac{1} 
{\epsilon_{t+1}} - 1 } \\[6pt]
\end{align}
$$
- $(4b)$: Likewise for the negative weight $e^{w_{t+1}}$,
$$
e^{w_{t+1}}
= \frac{1}{\sqrt{ \frac{1} 
{\epsilon_{t+1}} - 1 }}
$$

---
### Continuing the Inductive Argument
Hence,
$$
\begin{align}
&= \frac{1}{\sqrt{ \frac{1}{\epsilon_{t+1}} - 1 }}
(1 - \epsilon_{t+1})
+ \sqrt{ \frac{1}{\epsilon_{t+1} - 1} } (\epsilon_{t+1}) \\[6pt]

&= \sqrt{  
\frac{\epsilon_{t+1}}{1 - \epsilon_{t+1}} }
(1 - \epsilon_{t+1})
+ \sqrt{ \frac{1 - \epsilon_{t+1}}{\epsilon_{t+1}} }
(\epsilon_{t+1}) \\[6pt]

&= 2 \sqrt{ \epsilon_{t+1} (1 - \epsilon_{t+1}) }
\end{align}
$$

---
### Setting the Upper Bound
Recall that we assumed 
$$
\epsilon_{t+1} \leq \frac{1}{2} - \gamma
$$
Since $g(a) = a(1-a)$ is `monotonically increasing` in $\left[ 0, \frac{1}{2} \right]$,
$$
2 \ \sqrt{ \epsilon_{t+1} (1 - \epsilon_{t+1}) }
\leq \quad 2 \ \sqrt{ \left( \frac{1}{2} - \gamma \right) 
\left( \frac{1}{2} + \gamma \right)}
= \sqrt{ 1 - 4\gamma^2 }
$$

Using the inequality $1 - a \leq e^{-a}$, it follows that
$$
\begin{align}
\sqrt{ 1 - 4\gamma^2 }
&\leq e^{-4\gamma^2/2} \\[6pt]
&= e^{-2\gamma^2}
\end{align}
$$

Hence, we have proven that
$$
\boxed{\ Z_{T} \leq \exp(-2 \gamma^2 \ T) \ }
$$

---
## See Also
- [[AdaBoost]]
- [[AdaBoost Error Bound]]
- [[Maths behind AdaBoost]]