# Example-3

`Ques`: Solve $\begin{bmatrix}2 & 6 & 6 \\ 3 & 5 & 12 \\ 6 & 6 & 12 \end{bmatrix} \begin{bmatrix}x_{1} \\ x_{2} \\ x_{3} \end{bmatrix} = \begin{bmatrix} 20 \\ 25 \\ 30 \end{bmatrix}$

`Soln`:

`Step-1`
Since we want the pivot to be larger or equal to its terms below it, form the permutation matrices for each pivot that swap the rows with largest terms.

Set the permutation $P_{1} = \begin{bmatrix} 0 & 0 & 1 \\ 0 & 1 & 0 \\ 1 & 0 & 0\end{bmatrix}$ to switch rows $1$ and $3$
$$
\begin{align}
P_{1}A
&= \begin{bmatrix}
0 & 0 & 1 \\
0 & 1 & 0 \\
1 & 0 & 0
\end{bmatrix}
\begin{bmatrix}
2 & 6 & 6  \\
3 & 5 & 12 \\
2 & 6 & 6 
\end{bmatrix} \\[6pt]
 
&= 
\begin{bmatrix}
6 & 6 & 12 \\
3 & 5 & 12 \\
2 & 6 & 6
\end{bmatrix}
\end{align}
$$
Hence, $L_{1} = \begin{bmatrix}1 & 0 & 0 \\ -\frac{1}{2} & 1 & 0 \\ -\frac{1}{3}  & 0 & 1\end{bmatrix}$

$$
\begin{align}
L_{1} (P_{1}A) 
&= \begin{bmatrix}
1 & 0 & 0  \\
-\frac{1}{2} & 1 & 0 \\
-\frac{1}{3} & 0 & 1
\end{bmatrix}
\begin{bmatrix}
6 & 6 & 12 \\
3 & 5 & 12  \\
2 & 6 & 6
\end{bmatrix} \\[6pt]
&= \begin{bmatrix}
6 & 6 & 12 \\
0 & 2 & 6 \\
0 & 4 & 2
\end{bmatrix}
\end{align}
$$

``
`Step-2`
Now we set the permutation matrix $P_{2} = \begin{bmatrix}1 & 0 & 0 \\ 0 & 0 & 1 \\ 0 & 1 & 0 \end{bmatrix}$ to switch $2nd$ and $3rd$ rows.

$$
\begin{align}
P_{2} (L_{1} P_{1} A)

&= \begin{bmatrix}
1 & 0 & 0 \\
0 & 0 & 1  \\
0 & 1 & 0
\end{bmatrix}
\begin{bmatrix}
6 & 6 & 12 \\
0 & 2 & 6  \\
0 & 4 & 2
\end{bmatrix} \\[6pt]

&= \begin{bmatrix}
6 & 6 & 12 \\
0 & 4 & 2 \\
0 & 2 & 6
\end{bmatrix}
\end{align}
$$

Hence, $L_{2} = \begin{bmatrix}1 & 0 & 0 \\  0 & 1 & 0 \\ 0 & -\frac{1}{2} & 1\end{bmatrix}$

Thus,
$$
\begin{align}
L_{2} \ (P_{2} L_{1} P_{1} A)
&= \begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & -\frac{1}{2} & 1
\end{bmatrix}
\begin{bmatrix}
6 & 6 & 12 \\
0 & 4 & 2 \\
0 & 2 & 6
\end{bmatrix} \\[6pt]
&=
\begin{bmatrix}
6 & 6 & 12  \\
0 & 4 & 2 \\
0 & 0 & 5
\end{bmatrix}
\end{align}
$$

Hence, upper-triangular $U = \begin{bmatrix}6 & 6 & 12 \\ 0 & 4 & 2 \\ 0 & 0 & 5\end{bmatrix}$

`Step-3`
Right now, we have $L_{2} P_{2} \ L_{1} P_{1} \ A$ but we want it to be in $L_{2} \hat{L}_{1} \ P_{2} P_{1} \ A$
To achieve that, we will multiply the original term by $P_{n} P_{n}$ after $L_{n-1}$ where $n = 2 \to max(n) - 1$.

So here, we will multiply $L_{2} P_{2} \ L_{1} P_{1} \ A$ by $P_{2}P_{2}$, getting $L_{2} P_{2} \ L_{1} \ P_{2}P_{2} \  P_{1} \ A$.

Note that
- Inverse of permutation matrix is itself.
  So, when we do $P_{i} P_{i}$, we get $I$.
- When we pre-multiply by a permutation matrix $P$, we swap rows.
  When we post-multiply by a permutation matrix $P$, we swap columns.

$$
\begin{align}
P_{2} L_{1} P_{2}
&= \begin{bmatrix}
1 & 0 & 0  \\
0 & 0 & 1 \\
0 & 1 & 0
\end{bmatrix}
\begin{bmatrix}
1 & 0 & 0 \\
-\frac{1}{2} & 1 & 0 \\
-\frac{1}{3} & 0 & 1
\end{bmatrix}
\begin{bmatrix}
1 & 0 & 0 \\
0 & 0 & 1 \\
0 & 1 & 0 
\end{bmatrix} \\[6pt]

&= \begin{bmatrix}
1 & 0 & 0  \\
-\frac{1}{3} & 0 & 1 \\
-\frac{1}{2} & 1 & 0
\end{bmatrix} 
\begin{bmatrix}
1 & 0 & 0 \\
0 & 0 & 1 \\
0 & 1 & 0
\end{bmatrix} \\[6pt]

&= \begin{bmatrix}
1 & 0 & 0 \\
-\frac{1}{3} & 1 & 0 \\
-\frac{1}{2} & 0 & 1
\end{bmatrix} \\[6pt]

&= \hat{L}_{1}
\end{align}
$$

So, we get
$$
\begin{align}
L_{2} \hat{L}_{1} \ P_{2} P_{1} \ A &= U \\[6pt]
P_{2} P_{1} A &= \hat{L}_{1}^{-1} \hat{L}_{2}^{-1} \ U \\[6pt]
PA &= LU
\end{align}
$$


`Step-4`
So, the `permutation matrix` is
$$
\begin{align}
P &= \begin{bmatrix}
1 & 0 & 0 \\
0 & 0 & 1 \\
0 & 1 & 0
\end{bmatrix}
\begin{bmatrix}
0 & 0 & 1 \\
0 & 1 & 0 \\
1 & 0 & 0
\end{bmatrix} \\[6pt]

&= \begin{bmatrix}
0 & 0 & 1 \\
1 & 0 & 0 \\
0 & 1 & 0
\end{bmatrix}
\end{align}
$$

And the `lower triangular matrix` is 
$$
\begin{align}
L &= \hat{L}_{1}^{-1} \hat{L}_{2}^{-1} \\[6pt]
&= \begin{bmatrix}
1 & 0 & 0 \\
\frac{1}{3} & 1 & 0 \\
\frac{1}{2} & 0 & 1
\end{bmatrix}
\begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & \frac{1}{2} & 1
\end{bmatrix} \\[6pt]

&= \begin{bmatrix}
1 & 0 & 0 \\
\frac{1}{3} & 1 & 0 \\
\frac{1}{2} & \frac{1}{2} & 1
\end{bmatrix}
\end{align}
$$

Now, we have 
$$
\begin{align} 
&A\vec{x} = \vec{b}  \\
&\iff PA\vec{x} = P\vec{b} \\
&\iff LU\vec{x} = P\vec{b} \\
\end{align}
$$

where, 
$$
\begin{align}
P \vec{b}
&= \begin{bmatrix}
0 & 0 & 1 \\
1 & 0 & 0 \\
0 & 1 & 0
\end{bmatrix}
\begin{bmatrix}
20 \\ 25 \\ 30 
\end{bmatrix} \\[6pt]
&=
\begin{bmatrix}
30 \\ 20 \\ 25
\end{bmatrix}
\end{align}
$$

Let $U\vec{x} = \vec{d}$.
We solve $\underbrace{L \vec{d} = P \vec{b}}_{\text{Forward Substitution}} for \ \ \vec{d}$ and $\underbrace{U \vec{x} = \vec{d}}_{\text{Backward Substitution}} for \ \ \vec{x}$

Solving the `Forward Substitution`,
$$
\begin{align}
L\vec{d} &= P\vec{b} \\[6pt]

\begin{bmatrix}
1 & 0 & 0 \\
\frac{1}{3} & 1 & 0 \\
\frac{1}{2} & \frac{1}{2} & 1
\end{bmatrix}
\begin{bmatrix}
d_{1} \\
d_{2} \\
d_{3}
\end{bmatrix}
&= 
\begin{bmatrix}
30 \\
20 \\
25
\end{bmatrix} \\[6pt]

d_{1} &= 30 \\
d_{2} &= 10 \\
d_{3} &= 5 \\
\end{align}
$$
Hence, we get $\vec{d} = \begin{bmatrix}30 \\ 10 \\ 5\end{bmatrix}$.

Solving the `Backward Substitution`,
$$
\begin{align}
U\vec{x} &= \vec{d}  \\[6pt]
\begin{bmatrix}
6 & 6 & 12 \\
0 & 4 & 2 \\
0 & 0 & 5
\end{bmatrix}
\begin{bmatrix}
x_{1} \\
x_{2} \\
x_{3}
\end{bmatrix}
&= \begin{bmatrix}
30 \\ 10 \\ 5
\end{bmatrix} \\[6pt]

x_{3} &= 1 \\
x_{2} &= 2 \\
x_{1} &= 1
\end{align}
$$
Thus, we get $\vec{x} = \begin{bmatrix}1 \\ 2 \\ 1 \end{bmatrix}$.
