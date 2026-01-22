# AdaBoost R2 
#ml/ensemble/boosting/adaboost/R2

`AdaBoost R2` is the [[Classical ML/Ensemble/Boosting/AdaBoost]] variant for `regression tasks`.

## Algorithm 

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

## See Also
- [[Classical ML/Ensemble/Boosting/AdaBoost]]
- [[SAMME]]
