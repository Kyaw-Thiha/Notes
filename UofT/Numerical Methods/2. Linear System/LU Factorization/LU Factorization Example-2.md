# Example-2

`Ques`: Given $A = \begin{bmatrix}1 & 1 & -1 \\ 1 & -2 & 3 \\ 2 & 3 & 1 \end{bmatrix}$ and $\vec{b} = \begin{bmatrix}4 \\ -6 \\ 7\end{bmatrix}$, solving using 
[[LU Factorization (Without Pivot)]].

Firstly, find the lower triangular $L_{1} = \begin{bmatrix}1 & 0 & 0 \\ -1 & 1 & 0 \\ -2 & 0 & 1 \end{bmatrix}$
Hence, 
$$
\begin{align}
L_{1}A

&= \begin{bmatrix}
1 & 0 & 0 \\
-1 & 1 & 0 \\
-2 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
1 & 1 & -1 \\
1 & -2 & 3 \\
2 & 3 & 1
\end{bmatrix} \\[6pt]

&= \begin{bmatrix}
1 & 1 & -1 \\
0 & -3 & 4 \\
0 & 1 & 3
\end{bmatrix}
\end{align}
$$

Secondly, find the lower triangular $L_{2} = \begin{bmatrix}1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & \frac{1}{3} & 1\end{bmatrix}$
Hence,
$$
\begin{align}
L_{2}(L_{1}A)

&=  \begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & \frac{1}{3} & 1
\end{bmatrix}
\begin{bmatrix}
1 & 1 & -1 \\
0 & -3 & 4 \\
0 & 1 & 3
\end{bmatrix} \\[6pt]

&= \begin{bmatrix}
1 & 1 & -1 \\
0 & -3 & 4 \\
0 & 0 & \frac{13}{3}
\end{bmatrix}
\end{align}
$$

Thus, the upper triangular $U = \begin{bmatrix} 1 & 1 & -1 \\ 0 & -3 & 4 \\ 0 & 0 & \frac{13}{3}\end{bmatrix}$

and lower triangular $L = L_{1}^{-1} + L_{2}^{-1} - I = \begin{bmatrix}1 & 0 & 0 \\ 1 & 1 & 0 \\ 2 & -\frac{1}{3} & 1\end{bmatrix}$

---

Hence to solve $LU\vec{x} = \vec{b}$, let $U\vec{x} = \vec{d}$.
Now, we solve $L\vec{d} = \vec{b}$ for $\vec{d}$.

$$
\begin{align}
&L\vec{d} = \vec{b} \\[6pt]
&
\begin{bmatrix}
1 & 0 & 0 \\
1 & 1 & 0 \\
2 & -\frac{1}{3} & 1
\end{bmatrix}
\begin{bmatrix}
d_{1} \\ d_{2} \\ d_{3}
\end{bmatrix}
= 
\begin{bmatrix}
4 \\ -6 \\ 7
\end{bmatrix} \\[12pt]

&d_{1} = 4 \\
&d_{2} = -10 \\
&d_{3} = -\frac{13}{3}
\end{align}
$$

Therefore, $\vec{d} = \begin{bmatrix}4 \\ -10 \\ -\frac{13}{3}\end{bmatrix}$

---

Now we solve $U\vec{x} = \vec{d}$ for $\vec{x}$.

$$
\begin{align}
&\begin{bmatrix}
1 & 1 & -1 \\
0 & -3 & 4 \\
0 & 0 & \frac{13}{3}
\end{bmatrix}
\begin{bmatrix}
x_{1} \\
x_{2} \\
x_{3}
\end{bmatrix}
=
\begin{bmatrix}
4 \\ -10 \\ -\frac{13}{3}
\end{bmatrix} \\[12pt]

&x_{3} = -1 \\
&x_{2} = 2 \\
&x_{1} = 1
\end{align} 
$$

Therefore, $\vec{x} = \begin{bmatrix}1 \\ 2 \\ -1 \end{bmatrix}$

---
