# AdaBoost
#ml/ensemble/boosting/adaboost

`AdaBoost` is a variant of [[Boosting]] that explicitly weigh on the loss.

![AdaBoost](https://ars.els-cdn.com/content/image/3-s2.0-B9780128177365000090-f09-18-9780128177365.jpg)

---
## Classical Technique

First, train the model $h_{k}(x)$ on the weighted data.
Then, compute its weighted [[Loss Function]].

$$
\epsilon_{k} = \sum^N_{i=1} w_{i}^{(k)} I(h_{k}(x_{i}) \neq y)
$$

Second, we use this error to compute the weight of the model $h_{k}(x)$ on the final prediction.
$$
\alpha_{k} = \frac{1}{2} \ \ln \left( \frac{1 - \epsilon_{k}}{\epsilon_{k}} \right)
$$
This means if the model learn well, smaller $e_{k}$ $\implies$ larger $\alpha_{k}$

Third, we increase weights for mis-predicted samples, and decrease weight for correctly predicted samples.
$$
w_{i}^{(k+1)}
= \frac{w_{t}^{(k)} \exp(-\alpha_{k} \ y_{i} \ h_{k}(x_{i}))}
{Z_{k}}
$$
where
- $\alpha_{k}$ is the `model weight`, computed based on error
- $Z_{k}$ is the `normalizing constant` (so weights sum to 0)

Note that $y_{i} = h_{k}(x_{i}) = -1 \ \text{or} \ 1$.
Hence, if the model is correct, $y_{i}h_{k}(x_{i}) = +1$ 
and if the model is wrong, $y_{i}h_{k}(x_{i}) = -1$ 

Then, the final model predicts by `majority vote`
$$
F(x) =  I \left[  argmax_{k} \sum_{k} \alpha_{k} \ h_{k}(x)   >  \ 0 \right]
$$

---
## SAMME (Multiclass AdaBoost)

In `SAMME`, each base model $h_{k}(x)$ predicts a `class label` and the error weight is defined as
$$
\alpha_{k} = \ln\left( \frac{1 - \epsilon_{k}}{\epsilon_{k}} \right) + \ln(K-1)
$$
where $K$ is the number of classes in `Classification`

Then, the final model predicts by `majority vote`
$$
F(x) = argmax_{k} \sum_{t} \alpha_{k} \ I[h_{k}(x) = k]
$$
---
## AdaBoost R2 (Regression)

First, find the max error $E_{max}^{(k)} = \max_{i} e_{i}^{(k)}$
Then, normalize the error values.
$$
\tilde{e}_{i}^{(k)} = \frac{e_{i}^{(k)}} {E_{max}^{(k)}}
$$
Then, compute weighted error $\epsilon_{k} = \sum^N_{i=1} w_{i}^{(k)} \tilde{e}_{i}^{(k)}$

Second, assign the model weight $\beta_{k}$
$$
\beta_{k} = \frac{\epsilon_{k}}{1 - \epsilon_{k}}
$$
Note that $\beta_{k} = \alpha_{k} = \ln\left(  \frac{1 - \epsilon_{k}}{\epsilon_{k}} \right)$

Third, update the dataset by increasing weights of samples with larger normalized error $\tilde{e}_{i}^{(k)}$.
$$
w_{i}^{(k+1)} 
= \frac{w_{i}^{(k)} \ \beta_{k}^{(1 - \tilde{e}_{i}^{k} )}}{Z_{k}}
$$
The final model predicts by `weighted median` of the outputs.
Note that we use `median` since it is robust to outliers.

---
