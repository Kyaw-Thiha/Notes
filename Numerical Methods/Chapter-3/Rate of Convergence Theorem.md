# Rate of Convergence Theorem
#numerical-methods/non-linear/rate-of-convergence-theorem 

`Definition`
For the [[Fixed Point Iteration (FPI)]] $x_{k+1} = g(x_{k})$,
if $g'(\hat{x}),\ g''(\hat{x}),\ \dots ,\ g^{p-1}(\hat{x}) = 0$ but $g^p(\hat{x}) \neq 0$, 
then we have `p-th order convergence`

`Proof`
$$
\begin{align}
x_{k+1} 
&= g(x_{k}) \\[6pt]
&= g(\tilde{x} + (x_{k} - \tilde{x})) \\[6pt] 
&= g(\tilde{x})  
+ (x_{k} - \tilde{x}) \ g'(\tilde{x})
+ \frac{(x_{k} - \tilde{x})^2}{2!} \ g''(\tilde{x})
+ \dots \\[6pt]
&+ \frac{(x_{k} - \tilde{x})^{p-1}}{(p-1)!} \ g^{p-1}(\tilde{x})
+ \underbrace{\frac{(x_{k} - \tilde{x})^{p}}{(p)!} \ g^{p}(\tilde{x})}_{\text{Remainder Term}}
\end{align}
$$

If we have $g'(\tilde{x}), g''(\tilde{x}), \dots, g^{p-1}(\tilde{x}) = 0$, then we get
$$x_{k+1} = \tilde{x} + \frac{(x_{k} - \tilde{x})^p} {p!} \ g^p(n_{k})$$

Recall that $g(\tilde{x}) = \tilde{x}$.
Rearranging the equation, we get
$$
\frac{x_{k+1} - \tilde{x}}{(x_{k} - \tilde{x})^p} = \frac{1}{p!} \ g^p(n_{k})
$$

As $k \to \infty$, $x_{k} \to \tilde{x}$ and $n_{k} \in [\tilde{x}, x_{k}] \to \tilde{x}$.
We can rewrite this as 
$$
\lim_{ x_{k} \to \tilde{x} } 
\frac{x_{k+1} - \tilde{x}}{(x_{k} - \tilde{x})^p} = \frac{1}{p!} \ g^p(\tilde{x})
$$

We can see that if $p^{th}$ derivative of $g(\tilde{x})$ is not zero, we get $p^{th}$ order convergence.

We can see that by using [[Fixed Point Methods (FPM)#Second Form|Second Form]] of [[Fixed Point Iteration (FPI)]], we can pick a $h(x)$ s.t the $p^{th}$ derivative of $g$ is not zero.

