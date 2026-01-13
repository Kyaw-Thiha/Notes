# PAC Learning
#ml/statistical-learning/erm-with-inductive-bias
#ml/statistical-learning/pac-learning

`Probably Approximately Correct (PAC) Learning` is a way to formally describe what it means for a machine-learning algorithm to "learn" something reliably.

In other words,
> A learning algorithm is good if it can find model that is **probably** close to the truth, using reasonable amount of data.

---
## Inductive Bias

Recall from [[Empirical Risk Minimization]] that it can lead to hypothesis that can overfit.

To prevent that, we can restrict the hypothesis space $\mathcal{H}$ using `inductive bias` (assumptions).

$$
ERM_{\mathcal{H}} \ (S) \in \arg \min_{h \in \mathcal{H}} L_{S}(h)
$$
---

## Finite Hypothesis Classes
The simplest restriction is to use a `finite hypothesis class`.
For sufficiently large data, a finite hypothesis class $\mathcal{H}$ will not overfit.

To analyze this class of hypotheses, we will make two assumptions.

### Assumptions
- `Realizability`
  There exists a hypothesis in our hypothesis class that minimizes the true risk.
$$
\exists \ h^* \in \mathcal{H} \text{ s.t. } 
L_{\mathcal{D}, f}(h^*) = 0
$$
	This implies that $L_{S}(h^*) = 0$.

- `i.i.d Data`
  The samples in the training data are independently and identically distributed $(i.i.d)$ $w.r.t$ $\mathcal{D}$. $(S \sim \mathcal{D}^m)$.
  As $m\to \infty$, the better $S$ represents true distribution $\mathcal{D}$, and the better we can learn from it.

---
## Quantifying Uncertainty

The true loss of the [[Empirical Risk Minimization|empirical risk minimizer]] $L_{\mathcal{D}, f}$ will depend on the samples $S$.
Since $S$ is randomly generated, the true loss will be a random variable.

$\therefore$ We cannot be $100\%$ confident that $h_{S} = h^*$.
Hence, we need to quantify the probability that we will find a classifier whose error rate is not too large.

---
`Definitions`

Let us define the following
- $\delta$: The probability of getting a dataset $S$ that `does not represent` $\mathcal{D}$.
- $(1 - \delta)$: Our `confidence parameter`. 
  The probability of getting a good dataset.
- $\epsilon$: The `accuracy parameter`.
  $L_{\mathcal{D}, f}(h_{S}) > \epsilon$ is a classifier with error that is too large.
  $L_{\mathcal{D}, f}(h_{S}) \leq \epsilon$ is considered a success of the learning algorithm.
- $S \mid_{x}$: `Domain samples` from our training set.

---
### Error Upper Bound and Minimum Sample Size

`Goal`: Upper bound the probability that the true loss of the dataset exceeds $\epsilon$.
$$
D^m( \ \{ S\mid_{x}: L_{D, f}(h_{S}) > \epsilon \} \ ) \leq \delta
$$

`Using Realizability Assumption`
- Let $\mathcal{H}_{B}  = \{ h \in \mathcal{H}: L_{D, f}(h) > \epsilon \}$ be the set of bad hypotheses.
- Let $M = \{ S |_{x}: \exists h \in \mathcal{H}_{B}, L_{S}(h) = 0 \}$ be the set of misleading datasets.

The `realizability assumption` means $L_{S}(h_{S}) = 0$.
So, $L_{\mathcal{D}, f}(h_{S}) > \epsilon$ can only happen if we got one of the bad datasets in $M$.
$(\text{i.e: for some } h\in \mathcal{H}_{B}, L_{S}(h) = 0)$.

Hence, we get that
$$
\begin{align}
&\{ S \mid_{x}: L_{\mathcal{D}, f} > \epsilon \} \subseteq M \\[6pt]
\implies &M = \cup_{h\in \mathcal{H}_{B}}
\{ S\mid_{x}: L_{S}(h) = 0 \}
\end{align}
$$
Thus,
$$
\begin{align}
D^m( \{ S \mid_{x}: L_{D,f}(h_{S}) >\epsilon \} )
&\leq D^m(M) \\[6pt]
&= D^m \Big( \cup_{h \in \mathcal{H}_{B}} \{ S\mid_{x} : L_{S}(h) = 0 \} \Big)
\end{align}
$$

---
`Union Bound`
To reduce the expression $D^m \Big( \cup_{h \in \mathcal{H}_{B}} \{ S\mid_{x} : L_{S}(h) = 0 \} \Big)$, 
we will apply 
- `union bound` for sets $A$ and $B$ and distribution $\mathcal{D}$
- $\mathcal{D}(A \cup B) \leq \mathcal{D}(A) + \mathcal{D}(B)$

$$
D^m \Big( \cup_{h \in \mathcal{H}_{B}} \{ S\mid_{x} : L_{S}(h) = 0 \} \Big)
\leq \sum_{h \in \mathcal{H}_{B}}
D^m(\{ S \mid_{x}: L_{S}(h) = 0 \})
$$

`Using i.i.d`
Consider one hypothesis:
$$
h \in \mathcal{H}_{B},\ L_{S}(h) = 0
\iff \forall i, \ h(x_{i}) = f(x_{i})
$$
Then,
$$
\begin{align}
&D^m(\{ S\mid_{x}: L_{S}(h) = 0 \}) \\[6pt]
&= D^m(\{ S\mid_{x}: \forall i, \ h(x_{i}) = f(x_{i}) \}) \\[6pt]
&= \prod^m_{i=1} D^m (\{ h(x_{i}) = f(x_{i}) \})
& & \text{by i.i.d assumption}
\end{align}
$$

---
`Upper Bound on True Error`
Recall that [[Risk Function]] is the probability of making an incorrect classification.
Hence, the probability of a correct classification is $1 - L_{\mathcal{D},f}$.

Therefore,
$$
\begin{align}
\mathcal{D}(\{ h(x_{i}) = y_{i} \})
&= 1 - L_{D, f}(h) \\[6pt]
\implies \mathcal{D}(\{ h(x_{i}) = y_{i} \}) 
 &\leq 1 - \epsilon  
& &  \text{since } h\in \mathcal{H}_{B} \implies L_{D,f}(h) < \epsilon
\\[6pt]

\implies  
\mathcal{D}^m (\{ S \mid_{x}: L_{S}(h) = 0 \})
&\leq (1 - \epsilon)^m  \\[6pt]

\implies  
\mathcal{D}^m (\{ S \mid_{x}: L_{S}(h) = 0 \})
&\leq e^{-\epsilon m}
& & \text{applying } 1 - \epsilon \leq e^{-\epsilon}
\\[6pt]

\implies  
\sum_{h\in \mathcal{H}_{B}} \mathcal{D}^m (\{ S \mid_{x}: L_{S}(h) = 0 \})
&\leq |\mathcal{H}_{b}| e^{-\epsilon m} \\[6pt]

\implies  
\sum_{h\in \mathcal{H}_{B}} \mathcal{D}^m (\{ S \mid_{x}: L_{S}(h) = 0 \})
&\leq |\mathcal{H}| e^{-\epsilon m}
\end{align}
$$

---
`Lower Bound on Sample Size`
`Goal`: Lower bound the sample size that the true loss of the dataset exceeds $\epsilon$.

Recall that we have the probability bound
$$
\mathcal{D}^m (\{ S \mid_{x}: L_{S}(h) = 0 \})
\leq |\mathcal{H}| e^{-\epsilon m}
$$

Since `PAC Learning` requires $\mathcal{D}^m (\{ S \mid_{x}: L_{S}(h) = 0 \}) \leq \delta$, 
we need to find the minimum dataset size $m$ such that
$$
|\mathcal{H}| e^{-\epsilon m} \leq \delta
$$
Using algebra,
$$
\begin{align}
|\mathcal{H}| e^{-\epsilon m} &\leq \delta \\[6pt]

e^{-\epsilon m} &\leq \frac{\delta}{|\mathcal{H}|}  
\\[6pt]

-\epsilon m &\leq \log\left( \frac{\delta}{|\mathcal{H}|} \right)
\\[6pt]

\epsilon m &\geq \log\left( \frac{|\mathcal{H}|}{\delta} \right)
\\[6pt]

m &\geq \frac{\log\left( \mathcal{H}/\delta \right)}{\epsilon}
\\[6pt]
\end{align}
$$

---
`Restating`
We have now shown that for a `finite hypothesis space` $\mathcal{H}$ that contains `ERM solution`, the probability of getting a bad $\text{i.i.d}$ dataset $\delta$ is
$$
\mathcal{D}^m (\{ S\mid_{x}: L_{D,f}(h_{S}) > \epsilon \})
\leq |\mathcal{H}| \ e^{-\epsilon m} \leq \delta
$$

`Corollary`
For accuracy parameter $\epsilon > 0$, $\delta \in [0,1]$ and integer $m$ that satisfies 
$$
m \geq \frac{\log (|\mathcal{H}|/\delta)}{\epsilon}
$$
then for any labelling function $f$ and distribution $\mathcal{D}$ where realizability assumption holds
$(\text{i.e: } \exists h \in \mathcal{H} \text{ such that } L_{D,f}(h) = 0 \text{ with probability } (1-\delta))$ 
over an $\text{i.i.d}$ sample $S$ of size $m$, then for any ERM hypothesis $h_{S}$, it holds
$$
L_{\mathcal{D}, f} (h_{S}) \leq \epsilon
$$

---
`Upper Bound`
Hence, note that we have achieved our upper bound at
$$
\boxed{ \ \mathcal{D}^m (\{ S\mid_{x}: L_{D,f}(h_{S}) > \epsilon \})
\leq |\mathcal{H}| e^{-\epsilon m} \ }
$$
and also gotten a minimum dataset size at
$$
\boxed{\ m \geq \frac{\log (|\mathcal{H}|/\delta)}{\epsilon} \ }
$$

---
## See Also
- [[Empirical Risk Minimization]]
- [[Risk Function]]