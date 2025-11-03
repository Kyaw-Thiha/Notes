# Expected Test Error Decomposition
[[Expected Test Error]] can be decomposed as 
$$
E_{X, Y, D}[(\hat{y}_{D}(x) - y)^2]
= \underbrace{E_{X} [ (\bar{\hat{y}} - \bar{y}(x))^2] }_{\text{Bias}^2}
+ \underbrace{E_{X, D}[(\hat{y}_{D}(x) - \bar{\hat{y}}(x))^2]}_{\text{Variance}}
+ \underbrace{E_{X, Y}[(\bar{y}(x) - y)^2]}_{\text{Noise}}
$$

---

First, recall that [[Expected Test Error]] is
$$
\begin{align}
E_{X, Y, D} [ \ L(\hat{y}_{D}(x), \ y) \ ] 
&= E_{X, Y, D}[ \ (\hat{y}_{D}(x) - y)^2 \ ]
\end{align}
$$
where
- $\hat{y}_{D}(x)$ is predicted value for dataset $D$
- $y$ is the ground truth
- $E[y(x)]$ is the [[Expected Target Value]]

---

Let $\bar{\hat{y}}(x)$ from [[Expected Test Error Given Dataset D#Expected Predictor|Expected Predictor]] be `weighted average over functions`
Then, `Expected Test Error` as
$$
\begin{align}
&E_{X, Y, D}[ \ (\hat{y}_{D}(x) - y)^2 \ ] \\[6pt]

&= E_{X, Y, D}[ \ (\hat{y}_{D}(x) - \bar{\hat{y}}(x) + \bar{\hat{y}}(x) - y)^2 \ ]  
& \text{by adding and subtracting } \bar{\hat{y}}(x) \\[6pt]

&= E_{X, Y, D}[ \ ( (\hat{y}_{D}(x) - \bar{\hat{y}}(x)) + (\bar{\hat{y}}(x) - y) )^2 \ ] \\[6pt]

&= E_{X, Y, D}[ \ (\hat{y}_{D}(x) - \bar{\hat{y}})^2 \ ] + E_{X, Y, D} [ \  (\bar{\hat{y}} - y)^2 \ ]   \\[3pt]
&+ 2E_{X, Y, D}[ \  (\hat{y}_{D}(x) - \bar{\hat{y}}(x)) (\bar{\hat{y}}(x) - y)  \ ] 
& \text{ by properties of expectation}
\\[6pt]

&= E_{X, D}[ \ (\hat{y}_{D}(x) - \bar{\hat{y}})^2 \ ] + E_{X, Y} [ \  (\bar{\hat{y}} - y)^2 \ ] + 2E_{X, Y, D}[ \  (\hat{y}_{D}(x) - \bar{\hat{y}}(x)) (\bar{\hat{y}}(x) - y)  \ ] 
\\[6pt]
\end{align}
$$

---
We are going to prove that the `last term` $E_{X, Y, D}[ \  (\hat{y}_{D}(x) - \bar{\hat{y}}(x)) (\bar{\hat{y}}(x) - y)\ ]$  equals $0$.

$$
\begin{align}
&E_{X, Y, D}[ \  (\hat{y}_{D}(x) - \bar{\hat{y}}(x)) (\bar{\hat{y}}(x) - y)\ ] \\[6pt]
&= E_{X, Y}[ \ E_{D} [ \ (\hat{y}_{D}(x) - \bar{\hat{y}}(x) \ ] \  (\bar{\hat{y}}(x) - y)\ ] \\[6pt]

&= E_{X, Y}[ ( \ E_{D} [\hat{y}_{D}(x)] - \bar{\hat{y}}(x) \ ) \  (\bar{\hat{y}}(x) - y)\ ] \\[6pt]

&= E_{X, Y}[ \ ( \bar{\hat{y}}(x) - \bar{\hat{y}}(x) ) \  (\bar{\hat{y}}(x) - y)\ ] 
&\text{ since } \bar{\hat{y}}(x) = E_{D}[\hat{y}_{D}(x)]  \\[6pt]

&= 0
\end{align}
$$

---

Note that the `first term` is `Variance`
$$
E_{X, Y, D}[(\hat{y}_{D}(x) - y)^2] 
= \underbrace{E_{X, D}[(\hat{y}_{D}(x) - \hat{y}(x))^2]}_{\text{Variance}} 
+ E_{X, Y}[(\bar{\hat{y}}(x) - y)^2]
$$

So, decomposing the `second term` $E_{X, Y}[(\bar{\hat{y}}(x) - y)^2]$, we get
$$
\begin{align}
&E_{X, Y}[(\bar{\hat{y}} - y)^2] \\[6pt]

&= E_{X, Y}[ ( (\bar{\hat{y}}(x) - \bar{y}(x)) + (\bar{y}(x) - y) ) ^2] \\[6pt] 

&= E_{X, Y}[(\bar{\hat{y}} - \bar{y}(x))^2] + E_{X, Y}[(\bar{y}(x) - y))^2] + 2 E_{X, Y}[ \ (\bar{\hat{y}}(x) - \bar{y}(x))(\bar{y}(x) - y) \ ] \\[6pt] 

&= \underbrace{E_{X}[(\bar{\hat{y}} - \bar{y}(x))^2]}_{\text{Bias}^2}
+ \underbrace{E_{X, Y}[(\bar{y}(x) - y))^2]}_{\text{Noise}}
+ 2 E_{X, Y}[ \ (\bar{\hat{y}}(x)  
- \bar{y}(x))(\bar{y}(x) - y) \ ] \\[6pt] 
\end{align}
$$
where $\bar{y}$ is the [[Expected Target Value]]


We can further prove that the `third term` from this decomposition $E_{X, Y}[(\bar{\hat{y}}(x) - \hat{y}(x))(\hat{y}(x) - y)]$ equals $0$.

$$
\begin{align}
&E_{X, Y}[(\bar{\hat{y}} - \bar{y}(x))(\bar{y}(x) - y)] \\[6pt]

&= E_{X}[ \ (\bar{\hat{y}} - \bar{y}(x)) \ E_{Y|X}[(\bar{y}(x) - y)] \ ] \\[6pt] 

&= E_{X}[ \ E_{Y|X}[(\bar{y}(x) - y)] \ (\bar{\hat{y}} - \bar{y}(x)) \ ] \\[6pt] 

&= E_{X}[ \ (\bar{y}(x) - E_{Y|X}[y]) \ (\bar{\hat{y}} - \bar{y}(x)) \ ] \\[6pt] 

&= E_{X}[ \ (\bar{y}(x) - \bar{y}(x)) \ (\bar{\hat{y}} - \bar{y}(x)) \ ] \\[6pt] 

&= 0
\end{align}
$$

---

Hence, [[Expected Test Error]] can be decomposed as 
$$
E_{X, Y, D}[(\hat{y}_{D}(x) - y)^2]
= \underbrace{E_{X} [ (\bar{\hat{y}} - \bar{y}(x))^2] }_{\text{Bias}^2}
+ \underbrace{E_{X, D}[(\hat{y}_{D}(x) - \bar{\hat{y}}(x))^2]}_{\text{Variance}}
+ \underbrace{E_{X, Y}[(\bar{y}(x) - y)^2]}_{\text{Noise}}
$$