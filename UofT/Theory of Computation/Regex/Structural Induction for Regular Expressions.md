# Structural Induction

Let 
`Case-1`: $R = \emptyset$
$$
\begin{align}
L(R') &= \text{Ins0}(L(R)) \\
 &= \text{Ins0}(\emptyset) \\
 &= \emptyset \\
\end{align}
$$
Hence, $R' = \emptyset$

`Case-2`: $R = \epsilon$
$$
\begin{align}
L(R') &= \text{Ins0}(L(R)) \\
 &= \text{Ins0}(\epsilon) \\
 &= \{ 0 \} \\
\end{align}
$$
Hence, $R' = 0$

`Case-3`: $R = b$
$$
\begin{align}
L(R') &= \text{Ins0}(L(R)) \\
 &= \text{Ins0}(b) \\
 &= \{ b0, 0b\} \\
\end{align}
$$
Hence, $R' = ( b0 + 0b )$