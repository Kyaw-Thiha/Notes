# LU Factorization 

Note that when we carry out [[LU Factorization (Without Pivot)]], we divide the terms by the diagonal term to get the multipliers. 
But what if that diagonal term is 0 or close to 0?

Thats where `pivoting` comes into play.
`Pivoting` ensures that the diagonal term is always greater or equal to the terms below it.

This is achieved by swapping rows through multiplying by [[Permutation Matrix]] $P$.

Hence, 
$$
\begin{align}
&L_{m-1} P_{m-1} \dots L_{2} P_{2} L_{1} P_{1} \ A = U \\[6pt]

&\iff L_{m-1} \hat{L}_{m-2} \dots \hat{L}_{1} \ P_{m-1} \dots P_{1} \ A = U \\[6pt]

&\iff \underbrace{P_{m-1} \dots P_{1}}_{P} \ A = \underbrace{\hat{L}_{1}^{-1} \hat{L}_{2}^{-1} \dots L_{m-1}^{-1}}_{L} \ U \\[6pt]

&\iff PA = LU
\end{align}
$$

Thus, we get 
$$
\begin{align} 
&A\vec{x} = \vec{b}  \\
&\iff PA\vec{x} = P\vec{b} \\
&\iff LU\vec{x} = P\vec{b} \\
\end{align}
$$

Let $U\vec{x} = \vec{d}$.
We solve $\underbrace{L \vec{d} = P \vec{b}}_{\text{Forward Substitution}} for \ \ \vec{d}$ and $\underbrace{U \vec{x} = \vec{d}}_{\text{Backward Substitution}} for \ \ \vec{x}$

## Example
To see it in action, check the [[LU Factorization Example-3|Example]].
