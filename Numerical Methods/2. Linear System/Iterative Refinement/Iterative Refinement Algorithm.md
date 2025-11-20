# Iterative Refinement Algorithm

Compute $\hat{X}^{(0)}$ by solving $Ax = b$ in the [[Floating Points|Floating Point System]].

For $k = 0, 1, 2, \dots$ until solution is "good enough" or convergence failure, carry out
1. Compute $r^{(k)} = b - A\hat{x}^{(k)}$
2. Solve $Az^{(k)} = r^{(k)}$ for $\hat{z}^{(k)}$
3. Update $\hat{x}^{k+1} = \hat{x}^{k} + \hat{z}^{(k)}$


Note that the `step-2` is expensive.
But you can factor $PA = LU$ once only
And then use the factorization for each right hand side $r^{(k)}$
