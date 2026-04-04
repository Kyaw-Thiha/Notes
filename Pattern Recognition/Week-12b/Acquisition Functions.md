# Acquisition Functions for Active Learning
#ml/active-learning/acquisition-function

1. **Novelty Seeking**: $x^{*}_{t} = \arg\min_{x \in \mathcal{X}} p(x \mid S_{t-1})$
   Think of exploratory robots 
2. **Uncertainty Sampling**:

$$
x^{*}_{t} = \arg\max_{x \in \mathcal{X}}
- \sum_{i} P_{\theta}(y_{i} \mid x)
\log P_{\theta}(y_{i} \mid x)
$$
3. **Query by Committee**:
   Maintain models $h_{1}, \dots, h_{n}$, select $x$ with maximum disagreement amongst $h_{i}$.
   $(\text{This can be thought of as entropy over labellings})$.
4. **Expected Model Change**:
$$
x^{*} = \arg \max_{x \in \mathcal{X}} \sum_{i} 
P_{\theta}(y_{i} \mid x) || \ \nabla_{\theta}
(S \cup (x, y_{i})) \ ||
$$
   Find me the input in input space which would change the parameters of the network the most
5. **Expected Error Reduction**: 

$$
x^{*} = \arg \max_{x \in \mathcal{X}} \mathbb{E}_{P_\theta(y \mid x)} [ \ H(\theta \mid y) - H(\theta) \ ]
$$
7. **Variance Reduction**: A-Optimal Design, minimize the trace of inverse Fisher Information matrix.
$$
\mathcal{I}(\theta)
= N \int_{X} p(x) \int_{y} P_{\theta}(y \mid x)
\frac{\partial^{2}}{\partial \ \theta^{2}} 
\log P_{\theta}(y \mid x)
$$

---
## See Also
- [Excellent Survey by Settles (2009)]