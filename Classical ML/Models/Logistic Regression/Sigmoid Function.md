# Sigmoid Function
#ml/models/classic/logistic-regression/sigmoid  
#math/sigmoid
The `sigmoid function` maps real line $(-\infty, \infty)$ to $(0,1)$
$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

![Sigmoid Function](https://media.licdn.com/dms/image/v2/D4D12AQGIXdSG7IJCNw/article-cover_image-shrink_600_2000/article-cover_image-shrink_600_2000/0/1694183259537?e=2147483647&v=beta&t=lJ_qEzot0iGYhNpez9XGRNHjS-CDKHn3Wj-6iCQxRO0)


---
## Derivative
The derivative of `sigmoid function` is very convenient for `back-propagation`.
$$
\sigma(x)^{'} = \sigma(x) (1 - \sigma(x))
$$

![Sigmoid Function Derivative](https://storage.googleapis.com/lds-media/images/sigmoid-function-with-derivative.width-1200.png)

Let $\sigma(x) = \frac{1}{1 + e^{-x}}$
Then,
$$
\begin{align}
\frac{d}{dx} \sigma(x) 
&= \frac{d}{dx} \frac{1}{1 + e^{-x}} \\[6pt]
\sigma^{'}(x) &= -(1 + e^{-x})^{-2} \frac{d}{dx} (1 + e^{-x}) \\[6pt]
\sigma^{'}(x) &= (1 + e^{-x})^{-2} \ e^{-x} \\[6pt]

\sigma^{'}(x) &= \frac{1}{1 + e^{-x}} \times \frac{e^{-x}}{1 + e^{-x}} \\[6pt]

\sigma^{'}(x) &= \frac{1}{1 + e^{-x}} \times \frac{1 + e^{-x} - 1}{1 + e^{-x}} \\[6pt]

\sigma^{'}(x) &= \left( \frac{1}{1 + e^{-x}} \right) \left(1 - \frac{1}{1 + e^{-x}} \right) \\[6pt]

\sigma^{'}(x) &= \sigma(x) (1 - \sigma(x)) \\[6pt]
\end{align}
$$
---
## Inverse
The `inverse of sigmoid function` is
$$
\sigma^{-1}(x) = \log\left( \frac{\sigma(x)}{1 - \sigma(x)} \right)
$$
![Sigmoid Function Inverse](https://i.sstatic.net/3302q.png)

$$
\begin{align}
\text{Let } f(x) &= \frac{1}{1 + e^{-x}} \\[6pt]
y &= \frac{1}{1 + e^{-x}} \\[6pt]
\frac{1}{y} &= 1 + e^{-x} \\[6pt]
\frac{1}{y} - 1 &= e^{-x} \\[6pt]
\ln(\frac{1 - y}{y}) &= -x \\[6pt]
x &= \ln(\frac{y}{1 - y}) \\[6pt]
f^{-1}(x) &= \ln(\frac{x}{1 - x}) \\[6pt]
\end{align}
$$

---
## See Also
- [[Logistic Regression]]