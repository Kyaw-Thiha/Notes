# Fixed Point Methods (FPM)
#numerical-methods/non-linear/fpm 

- `Root Finding Problem`: $F(\tilde{x}) = 0$ 
- `Fixed Point Problem`: $\tilde{x} = g(\tilde{x})$

Note that $F(\tilde{x}) = 0 \iff \tilde{x} = g(\tilde{x})$
E.g.: $\underbrace{x - e^{-x} = 0}_{\text{Root finding problem}} \iff \underbrace{x = e^{-x}}_{\text{Fixed point problem}}$

We can rewrite as
- `First Form`: $g(\tilde{x}) = \tilde{x} - F(\tilde{x})$
- `Second Form`: $g(\tilde{x}) = \tilde{x} - h(\tilde{x})F(\tilde{x})$, where $h(\tilde{x})$ is an `auxilliary function`

## First Form

$$
g(\tilde{x}) = \tilde{x} = F(\tilde{x})
$$

If we use the `First Form`, then $F(\tilde{x}) = 0 \iff \tilde{x} = g(\tilde{x})$ 
`Proof`:
LHS: $F(\tilde{x}) = 0$
RHS: 
$$
\begin{align}
\tilde{x} &= g(\tilde{x}) \\[6pt]
\tilde{x} &= \tilde{x} - F(\tilde{x}) \\[6pt]
0 &= - F(\tilde{x}) \\[6pt]
F(\tilde{x}) &= 0 \\[6pt]
\end{align}
$$
Hence, LHS = RHS

## Second Form
$$
g(\tilde{x}) = \tilde{x} - h(\tilde{x})F(\tilde{x})
$$

In `Second Form`, if $F(\tilde{x}) = 0$, then $\tilde{x} = g(\tilde{x})$.
However, we could have $\tilde{x} = g(\tilde{x})$ but $F(\tilde{x}) \neq 0$.
This situation occurs if $h(\tilde{x}) = 0$.
Hence, the two equations aren't equivalent.

Furthermore, after we find a fixed point, we need to check if it is a root or not.

The advantage of the `Second Form` is that there's flexibility in designing $g(\tilde{x})$, to make iteration converge faster.

## See Also
- [[Non-Linear Methods]]
