# Maths Behind EM Algorithm
#ml/classic-models/mixture-model/em-algorithm/maths  

Let $m_{1:K}$ where $\sum^K_{j=1} m_{j} = 1$ be the `mixing probabilities`
Let $\psi_{1:K}$ where $\psi_{j} = \{ \mu_{j},\ C_{j} \}$ be the `Gaussian Likelihood parameters`
Let $\theta = \{ m_{1:K},\ \psi_{1:K} \}$ be the `model parameters`.

where
- $\mu_{j}$ is the `mean` of $j^{th}$ gaussian component
- $C_{j}$ is the `covariance matrix` of $j^{th}$ gaussian component

---
`MLE Estimate`

Based on `MLE Estimate` from [[Maths Behind GMM]], we get
$$
\begin{align}
L(\theta)
&= -\sum^N_{i=1} \log \sum^K_{j=1}  
m_{j} \ p(y_{i} \ | \ l=j, \theta) 
\end{align}
$$
where $\theta = \{ m_{1:K}, \mu_{1:K}, C_{1:K} \}$

with constraints of
$$
1. \begin{cases}
m_{j} \in [0, 1] \\[6pt]
\sum^K_{j=1} m_{j} = 1
\end{cases}
\ \quad  \
2. \ C_{j} \text{ is symmetric positive definite}
$$

---
`Lagrangian`

Since we have a constraint of $\sum^K_{j=1} m_{j} = 1$, we can convert it into `unconstrained optimization problem` using [[Lagrange Multipliers]] with $g(x) = \sum^K_{j=1} m_{j} - 1$.

Note that 
- $m_{j} \in [0,1]$ is implicitly constrained by $g(x) = \sum^K_{j=1} m_{j} - 1$
- $C_{j}$ is always `symmetric positive definite` by property of covariance matrix

Hence, we can define the `Lagrangian` as
$$
\begin{align}
L(\theta, \lambda)

&= - \sum^N_{i=1} \log  
\sum^K_{j=1} m_{j} \ p(y_{i} \ | \ l=j, \theta)
+ \lambda \left( \sum^K_{j=1} m_{j} - 1 \right) 
\\[6pt]

&= - \sum^N_{i=1} \log  
\sum^K_{j=1} m_{j} \ p(y_{i} \ | \ \psi_{j})
+ \lambda \left( \sum^K_{j=1} m_{j} - 1 \right) 
\\[6pt]
\end{align}
$$

---
`Optimizing`

$$
L(\theta, \lambda) 
= - \sum^N_{i=1} \log  
\sum^K_{j=1} m_{j} \ p(y_{i} \ | \ \psi_{j})
+ \lambda \left( \sum^K_{j=1} m_{j} - 1 \right) 
$$

To optimize it, we need to find the `partial derivatives`
$$
\frac{\partial L}{\partial \lambda} = 0
, \ \quad
\frac{\partial L}{\partial m_{j}} = 0
, \ \quad
\frac{\partial L}{\partial \psi_{j}} = 0
$$

---
`W.r.t lambda`
The constraint is adhered as per [[Lagrange Multipliers]].
$$
\begin{align}
\frac{\partial L}{\partial \lambda} &= 0 \\
\sum^K_{j=1} m_{j} - 1 & = 0 \\
\sum^K_{j=1} m_{j}  & = 1
\end{align}
$$

---
`W.r.t m_j`
$$
L(\theta, \lambda) 
= - \sum^N_{i=1} \log  
\sum^K_{j=1} m_{j} \ p(y_{i} \ | \ \psi_{j})
+ \lambda \left( \sum^K_{j=1} m_{j} - 1 \right) 
$$
Differentiating it, we get
$$
\begin{align}
&\frac{\partial L}{\partial m_{j}} \\[6pt]
&= -\sum^N_{i=1} \frac{1}{\sum^K_{h=1} m_{h} \ p(y_{i} | \psi_{h})}
\frac{\partial}{\partial m_{j}}
\left[ m_{j} p(y_{i}|\psi_{j})  
+ \sum_{h\neq j} m_{h} \ p(y_{i} | \psi_{h}) \right]
+ \lambda \\[6pt]

&= -\sum^N_{i=1} \frac{1}{\sum^K_{h=1} m_{h} \ p(y_{i} | \psi_{h})}
\frac{\partial}{\partial m_{j}}
\left[ m_{j} p(y_{i}|\psi_{j}) \right]
+ \lambda \\[6pt]

&= -\sum^N_{i=1} \frac{1}{\sum^K_{h=1} m_{h} \ p(y_{i} | \psi_{h})}
p(y_{i}|\psi_{j}) + \lambda \\[6pt]
\end{align}
$$

Recall from [[Maths Behind GMM]] that we defined `posterior (responsibilites)` as $\gamma_{i,j} = \frac{m_{j} \ p(y_{i} | l=j,\theta)}{p(y_{i} | \theta)}$
$$
\begin{align}
\gamma_{i,j}  
&= \frac{m_{j} \ p(y_{i} | l=j,\theta)}{p(y_{i} | \theta)} \\[6pt]

\frac{\gamma_{i,j}}{m_{j}}
&= \frac{p(y_{i} | \psi_{j})}{\sum^K_{h=1} m_{h} \ p(y_{i} | \psi_{h})} \\[6pt]

\end{align}
$$

Hence, we can substitute it into our differentiated solution
$$
\begin{align}
\frac{\partial L}{\partial m_{j}}
&= 0 \\[6pt]

-\sum^N_{i=1} \frac{p(y_{i}|\psi_{j})} 
{\sum^K_{h=1} m_{h} \ p(y_{i} | \psi_{h})}
 + \lambda &= 0 \\[6pt]

-\sum^N_{i=1} \frac{\gamma_{i,j}}{m_{j}} + \lambda
&= 0 \\[6pt]

m_{j} &= \frac{1}{\lambda} \sum^N_{i=1} \gamma_{i,j}
\end{align}
$$

Continuing on
$$
\begin{align}
&\begin{cases}
m_{j} = \frac{1}{\lambda} \sum^N_{i=1} \gamma_{i.j} \\[6pt]
\sum^K_{j=1} m_{j} = 1
\end{cases} \\[6pt]

&\implies  
\sum^K_{j=1} \frac{1}{\lambda} \  
\sum^N_{i=1} \gamma_{i,j} = 1 \\[6pt]

&\implies  
\frac{1}{\lambda} \sum^K_{j=1} \  
\sum^N_{i=1} \gamma_{i,j} = 1 \\[6pt]

&\implies  
\begin{cases}
\sum^N_{i=1} \gamma_{i,j} = 1 \\[6pt]
\frac{1}{\lambda} \sum^N_{i=1} 1 = 1
\end{cases} \\[6pt]

&\implies \lambda = N
\end{align}
$$

Hence, we have derived that 
$$
\boxed{m_{j} = \frac{1}{N} \sum^N_{i=1} \gamma_{i,j}}
$$

---
`W.r.t psi`
$$
L(\theta, \lambda) 
= - \sum^N_{i=1} \log  
\sum^K_{j=1} m_{j} \ p(y_{i} \ | \ \psi_{j})
+ \lambda \left( \sum^K_{j=1} m_{j} - 1 \right) 
$$
Differentiating it, we get
$$
\begin{align}
&\frac{\partial L}{\partial \psi_{j}} \\[6pt]

&= -\sum^N_{i=1} \frac{1}{\sum^K_{h=1} m_{h} \ p(y_{i} | \psi_{h})}
\frac{\partial}{\partial \psi_{j}}
\sum^K_{h=1} m_{h} p(y_{i} | \psi_{h})
\\[6pt]

&= -\sum^N_{i=1} \frac{1}{\sum^K_{h=1} m_{h} \ p(y_{i} | \psi_{h})}
\frac{\partial}{\partial \psi_{j}}
\left[ m_{j} \ p(y_{i}|\psi_{j})
+ \sum_{h \neq j} m_{h} \ p(y_{i} | \psi_{h}) 
\right]
\\[6pt]

&= -\sum^N_{i=1} \frac{1}{\sum^K_{h=1} m_{h} \ p(y_{i} | \psi_{h})}
\frac{\partial}{\partial \psi_{j}}
 m_{j} \ p(y_{i}|\psi_{j})
\\[6pt]

&= -\sum^N_{i=1} \frac{m_{j}}{\sum^K_{h=1} m_{h} \ p(y_{i} | \psi_{h})}
\frac{\partial}{\partial \psi_{j}}
p(y_{i}|\psi_{j})
\\[6pt]
\end{align}
$$

`Trick`: In order get a quadratic term
$$
\begin{align}
\frac{\partial}{\partial \psi}  
\log p(\psi)
&= \frac{1}{p(\psi)}
\frac{\partial}{\partial \psi} p(\psi) \\[6pt]

p(\psi) \ \frac{\partial}{\partial \psi}  
\log p(\psi)
&= \frac{\partial}{\partial \psi} p(\psi) \\[6pt]
\end{align}
$$

Hence,
$$
\begin{align}
\frac{\partial L}{\partial \psi_{j}}
&= -\sum^N_{i=1} \frac{m_{j}}{\sum^K_{h=1} m_{h} \ p(y_{i} | \psi_{h})}
\frac{\partial}{\partial \psi_{j}}
p(y_{i}|\psi_{j})
\\[6pt]

&= -\sum^N_{i=1} \frac{m_{j} \ p(y_{i} | \psi_{j})} 
{\sum^K_{h=1} m_{h} \ p(y_{i} | \psi_{h})}
\frac{\partial}{\partial \psi_{j}}
\log p(y_{i}|\psi_{j})
\\[6pt]

&= -\sum^N_{i=1} \gamma_{i,j} \
\frac{\partial}{\partial \psi_{j}}
\log p(y_{i}|\psi_{j})
\\[6pt]
\end{align}
$$

Since $p(y_{i} | \psi_{j})$ is `Gaussian` per [[Gaussian Mixture Model]],
$$
\begin{align}
p(y_{i} | \psi_{j})
&= \frac{1}{(2\pi)^{d/2} C^{1/2}}
\ \exp\left( -\frac{1}{2} (y_{i} - \mu_{j})^T C_{j}^{-1} (y_{i} - \mu_{j}) \right) \\[6pt]

\log p(y_{i} | \psi_{j})
&= \frac{d}{2} \log(2\pi)  
+ \frac{1}{2} \log|C_{j}|
 -\frac{1}{2} (y_{i} - \mu_{j})^T C_{j}^{-1} (y_{i} - \mu_{j}) 
\end{align}
$$

Hence, we can use [[MLE for Gaussian Distribution]] to continue.

Now, differentiating $w.r.t$ `mean` $\mu_{j}$,
$$
\begin{align}
\frac{\partial L}{\partial \mu_{j}} &= 0 
\\[6pt]

- \sum^N_{i=1} \gamma_{i,j} \ 
\frac{\partial}{\partial \mu_{j}}  
\log p(y_{i} \ | \ \mu_{j}, C_{j})  
&= 0 \\[6pt]

- \sum^N_{i=1} \gamma_{i,j} \ 
C^{-1} (y_{i} - \mu_{j})  
&= 0  \\[6pt]

\mu_{j}
&= \frac{\sum_{i} (\gamma_{i,j} \ y_{i})} 
{\sum_{i} \gamma_{i,j}}
\end{align}
$$

Now, differentiating $w.r.t$ `inverse covariance` $C_{j}^{-1}$,
$$
\begin{align}
\frac{\partial L}{\partial \mu_{j}} &= 0 
\\[6pt]

- \sum^N_{i=1} \gamma_{i,j} \ 
\frac{\partial}{\partial C_{j}^{-1}}  
\log p(y_{i} \ | \ \mu_{j}, C_{j})  
&= 0 \\[6pt]

- \sum^N_{i=1} \gamma_{i,j} \ 
\frac{\partial}{\partial C_{j}}  
\log p(y_{i} \ | \ \mu_{j}, C_{j})  
&= 0 \\[6pt]

- \sum^N_{i=1} \gamma_{i,j} \ 
\left( \frac{1}{2} (y_{i} - \mu_{j}) 
(y_{i} - \mu_{j})^T - \frac{1}{2} C_{j} \right)
&= 0 \\[6pt]

-\frac{1}{2} \sum^N_{i=1} \gamma_{i,j} \ 
(y_{i} - \mu_{j}) (y_{i} - \mu_{j})^T  
+ \frac{1}{2} C_{j} \sum^N_{i=1} \gamma_{i,j}
&= 0 \\[6pt]

C_{j} &= 
\frac{\sum_{i} \gamma_{i,j}  
(y_{i} - \mu_{j}) (y_{i} - \mu_{j})^T} 
{\sum_{i} \gamma_{i,j}}
\end{align}
$$

Hence, we have derived that
$$
\boxed{\mu_{j} = \frac{\sum_{i} \gamma_{i,j} \ y_{i}}{\sum_{i} \gamma_{i,j}}}
\quad \text{and} \quad
\boxed{C_{j} = 
\frac{\sum_{i} \gamma_{i,j}  
(y_{i} - \mu_{j}) (y_{i} - \mu_{j})^T} 
{\sum_{i} \gamma_{i,j}} }
$$

---
## See Also
- [[Maths Behind GMM]]
- [[EM Algorithm]]