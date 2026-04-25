# Example-1

`Ques`: Given $\begin{bmatrix}1 & 1 & 1 \\ 2 & 3 & 5 \\ 4 & 6 & 8\end{bmatrix}$, find $L$ and $U$.

`Soln`: 
Getting the first lower triangular $L_{1}$, 
$$
L_{1} 

= \begin{bmatrix}
1 & 0 & 0 \\
-\frac{2}{1} & 1 & 0 \\
-\frac{4}{1} & 0 & 1
\end{bmatrix}

= \begin{bmatrix}
1 & 0 & 0 \\
-2 & 1 & 0 \\
-4 & 0 & 1
\end{bmatrix}
$$
Then,
$$
\begin{align}
L_{1}A  
&= \begin{bmatrix}
1 & 0 & 0  \\
-2 & 1 & 0 \\
-4 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
1 & 1 & 1  \\
2 & 3 & 5 \\
4 & 6 & 8
\end{bmatrix} \\[6pt]

&= \begin{bmatrix}
1 & 1 & 1 \\
0 & 1 & 3 \\
0 & 2 & 4
\end{bmatrix}
\end{align}
$$

Next, getting the second lower-triangular $L_{2}$,
$$
L_{2} 

= \begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & -\frac{2}{1} & 1
\end{bmatrix}

= \begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & -2 & 1
\end{bmatrix}
$$
Then, 
$$
\begin{align}
L_{2}(L_{1}A)  
&=  \begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & -2 & 1 
\end{bmatrix}
\begin{bmatrix}
1 & 1 & 1 \\
0 & 1 & 3 \\
0 & 2 & 4
\end{bmatrix} \\[6pt]
 
&= \begin{bmatrix}
1 & 1 & 1  \\
0 & 1 & 3 \\
0 & 0 & -2
\end{bmatrix}
\end{align}
$$

Hence, the upper triangular $U = \begin{bmatrix} 1 & 1 & 1 \\ 0 & 1 & 3 \\ 0 & 0 & -2\end{bmatrix}$
and lower triangular $L=L_{1}^{-1} L_{2}^{-1}$ 
$$
\begin{align}
L &= \begin{bmatrix}
1 & 0 & 0  \\
2 & 1 & 0 \\
4 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 2 & 1
\end{bmatrix}
 \\[6pt]
&= \begin{bmatrix}
1 & 0 & 0 \\
2 & 1 & 0 \\
4 & 2 & 1
\end{bmatrix}
\end{align}
$$

Alternatively, 
$$
\begin{align}
L  
&= L_{1}^{-1} + L_{2}^{-1} - I \\[6pt]

&= \begin{bmatrix}
1 & 0 & 0  \\
2 & 1 & 0 \\
4 & 0 & 1
\end{bmatrix}
+
\begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 2 & 1
\end{bmatrix}
-
\begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}
 \\[6pt]
 
&= \begin{bmatrix}
2 & 0 & 0 \\
2 & 2 & 0 \\
4 & 2 & 2
\end{bmatrix}
-
\begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix} \\[6pt]

&= \begin{bmatrix}
1 & 0 & 0 \\
2 & 1 & 0 \\
4 & 2 & 1
\end{bmatrix}
\end{align}
$$
