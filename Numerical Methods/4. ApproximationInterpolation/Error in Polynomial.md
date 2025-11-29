# Error in Polynomial
#numerical-methods/interpolation/polynomial/error   

`Error in Polynomial` can be defined as 
$$
E(x) = Y(x) - P(x)
$$
where
- $Y(x)$ is the `Underlying Function`
- $P(x)$ is the `Interpolating Polynomial`

For a simple interpolation $P(x_{i}) = Y_{i}$, $i=0, 1, 2, \dots, n$, we can show that 
$$
E(x) = \frac{y^{(n+1)}}{(n+1)!} (\epsilon) \prod^n_{i=0} (x-x_{0})
$$
where
$$
\begin{align}
\epsilon &\in span\{ x_{0}, \dots,\ x_{n},\ x \}  
\\[6pt]
&= [min\{ x_{0}, \dots,\ x_{n},\ x \},  
max\{ x_{0}, \dots,\ x_{n}, \ x \}]  
\end{align}
$$

---
## See Also
- [[Polynomial Interpolation]]