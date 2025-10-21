# MLE for Gaussian Distribution

To learn `MLE`, we need to find the parameters $\mu$ and $\Sigma$ that results in best `Likelihood`. 

Given $\{ x_{i} \}^N_{i=1} \sim N(\mu, \Sigma)$.
$$
\begin{align}
P(D|\theta) &= P(x_{1}, \dots, x_{N}| \mu, \Sigma) \\
&= \Pi^N_{i=1} P(x_{i} | \mu, \Sigma) \\
&= \Pi^N_{i=1} \frac{1}{\sqrt{ (2\pi)^d |\Sigma| } } \exp\left( -\frac{1}{2} (x_{i} - \mu)^T \Sigma^{-1} (x_{i} - \mu) \right)
\end{align}
$$

To find the best parameters $\mu, \Sigma$
$$
\begin{align}
&argmin_{\mu, \Sigma} \ \Pi^N_{i=1} \frac{1}{\sqrt{ (2\pi)^d |\Sigma| } } \exp\left( -\frac{1}{2} (x_{i} - \mu)^T \Sigma^{-1} (x_{i} - \mu) \right)  \\

&= argmin_{\mu, \Sigma} -\ln \ \Pi^N_{i=1} \frac{1}{\sqrt{ (2\pi)^d |\Sigma| } } \exp\left( -\frac{1}{2} (x_{i} - \mu)^T \Sigma^{-1} (x_{i} - \mu) \right)  \\

&= argmin_{\mu, \Sigma} \sum^N_{i=1} \frac{1}{2} (x_{i} - \mu) ^T \Sigma^{-1} (x_{i} - \mu) - \ln\left( \frac{1}{|\Sigma|^{1/2}} . \frac{1}{(2\pi)^{d/2}}\right) \\

&= argmin_{\mu, \Sigma} \sum^N_{i=1} \frac{1}{2} (x_{i} - \mu) ^T \Sigma^{-1} (x_{i} - \mu) - \frac{N}{2}\ln|\Sigma| - \frac{Nd}{2}.\ln (2\pi) \\
\end{align}
$$



To minimize it, we need to differentiate the 2 `partial derivatives`
- First with respect to $\mu$
- Second with respect to $\Sigma$

### First Partial Derivative
Differentiating the first partial derivative,

$$
\begin{align}
\frac{\partial L}{\partial \mu} &= \sum^N_{i=1} \frac{1}{2} \frac{\partial}{\partial \mu} (x_{i} - \mu)^T \Sigma^{-1} (x_{i} - \mu)  \\
&= \sum^N_{i=1} \frac{\partial}{\partial \mu} \frac{1}{2}  \mu^T \Sigma^{-1} \mu - \mu^T.\Sigma^{-1}x_{i} + \frac{1}{2} x_{i}^T \Sigma^{-1} x_{i} \\
&= \sum^N_{i=1} \Sigma^{-1} \mu - x_{i}.\Sigma^{-1} \\
&= N \Sigma^{-1} \mu - \Sigma^{-1} \sum^N_{i=1} x_{i} \\
\end{align}
$$

Setting $\frac{\partial L}{\partial \mu} = 0$,
$$
\begin{align}
\frac{\partial L}{\partial \mu} &= 0 \\[6pt]
N \Sigma^{-1} \mu - \Sigma^{-1} \sum^N_{i=1} x_{i} &= 0 \\[6pt]
N \mu - \sum^N_{i=1} x_{i} &= 0 \\[6pt]
N \mu &= \sum^N_{i=1} x_{i} \\[6pt]
\mu &= \frac{1}{N} \sum^N_{i=1} x_{i} \\[6pt]
\end{align}
$$

### Second Partial Derivative
Now, let's differentiate the second partial derivative.
But first, will apply $\det(A^{-1}) = \frac{1}{\det(A)}$ to get 
$$
\begin{align}
\ln|\Sigma| &= \ln|(\Sigma^{-1})^{-1}| \\
&= \ln \frac{1}{|\Sigma^{-1}|} \\
&= -\ln|\Sigma^{-1}|
\end{align}
$$

So, instead of finding $\frac{\partial L}{\partial \Sigma} = 0$, we can find $\frac{\partial L}{\partial \Sigma^{-1}}$

Then we use these trace properties $\frac{\partial}{\partial A} tr(BA) = B^T$ and $\frac{\partial}{\partial A} \ln|A| = (A^{-1})^T$ .

So, differentiating the second part, we get
$$
\begin{align}
\frac{\partial L}{\partial \Sigma^{-1}}   
&= \sum^N_{i=1} \frac{1}{2} \frac{\partial}{\partial \Sigma^{-1}} (x_{i} - \mu)^T \Sigma^{-1} (x_{i} - \mu) - \frac{N}{2} \frac{\partial}{\partial \Sigma^{-1}} \ln|\Sigma^{-1}| \\

&= \sum^N_{i=1} \frac{1}{2} \frac{\partial}{\partial \Sigma^{-1}} tr((x_{i} - \mu)^T \Sigma^{-1} (x_{i} - \mu)) - \frac{N}{2} \Sigma &\text{by } \frac{\partial}{\partial A} \ln |A| = (A^{-1})^T \\

&= \sum^N_{i=1} \frac{1}{2} \frac{\partial}{\partial \Sigma^{-1}} tr((x_{i} - \mu)^T  (x_{i} - \mu) \Sigma^{-1}) - \frac{N}{2} \Sigma  &\text{by cyclic property of trace} \\

&= \sum^N_{i=1} \frac{1}{2} (x_{i} - \mu)(x_{i} - \mu)^T - \frac{N}{2} \Sigma &\text{by } \frac{\partial}{\partial A} tr(BA) = B^T  \\

\end{align}
$$

Setting $\frac{\partial L}{\partial \Sigma^{-1}} = 0$,
$$
\begin{align}
\frac{\partial L}{\partial \Sigma^{-1}} &= 0 \\
\sum^N_{i=1} \frac{1}{2} (x_{i} - \mu)(x_{i} - \mu)^T - \frac{N}{2} \Sigma &= 0 \\
\frac{N}{2}.\Sigma &= \sum^N_{i=1} \frac{1}{2} (x_{i} - \mu)(x_{i} - \mu)^T  \\
\Sigma &= \sum^N_{i=1} \frac{1}{N} (x_{i} - \mu)(x_{i} - \mu)^T  
\end{align}
$$

Hence, $\Sigma_{MLE} = \frac{1}{N} \sum^N_{i=1} (x_{i}-\mu)(x_{i} - \mu)^T$