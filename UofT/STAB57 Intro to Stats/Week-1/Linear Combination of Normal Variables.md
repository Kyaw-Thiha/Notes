# Linear Combinations of Normal Variables
- Let $X_{i} \sim N(\mu, \ \sigma^{2}_{i})$ where $i=1, 2, \dots, n$ and $X_{i}$'s are independent.
- Let $Y$ be a [[Linear Combination of Normal Variables|linear combination]] of all $X_{i}$'s with

$$
Y = a_{1}X_{1} + a_{2}X_{2} + \dots + a_{n}X_{n} + b
= \sum^{n}_{i=1} a_{i}X_{i} + b
$$
where $a_{1}, \ a_{2}, \ a_{n}, \ b$ are constants.

- Then,

$$
Y \sim N\left( \sum^{n}_{i=1} a_{i} \mu_{i} + b,  
\ \sum^{n}_{i=1} a_{i}^{2} \sigma_{i}^{2} \right)
$$

---
## Derivation
Using [[Moment Generating Function(MGF)|moment generating function]],
$$
\begin{align}
M_{Y}(t)
&= E[e^{ty}] \\[6pt]
&= E[e^{t(a_{1} x_{1} + a_{2} x_{2} + \dots +  
a_{n}x_{n} + b)}] \\[6pt]
&= E[e^{ta_{1} x_{1} \ + \ ta_{2} x_{2} \ + \ \dots
\ + \ ta_{n}x_{n} \ + \ tb}] \\[6pt]
&= E[e^{ta_{1}x_{1}} \cdot e^{ta_{2}x_{2}} \cdot \
\dots \ \cdot e^{ta_{n}x_{n}} \cdot e^{tb}] \\[6pt]
&= E[e^{ta_{1}x_{1}}] \cdot E[e^{ta_{2}x_{2}}] \cdot 
\ \dots \ \cdot E[e^{ta_{n}x_{n}}] \cdot E[e^{tb}] 
\\[6pt]
\end{align}
$$