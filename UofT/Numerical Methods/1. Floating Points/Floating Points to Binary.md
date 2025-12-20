# Floating Points to Binary
Remember that any number $x$ in the floating-point system is represented as:

$$
x = \pm \left( d_0 + \frac{d_1}{\beta} + \frac{d_2}{\beta^2}  + \cdots + \frac{d_{t-1}}{\beta^{(t-1)}}  \right) \, \beta^e
$$

where  

- $0 \leq d_i \leq \beta - 1,\; i = 0, \dots, t-1$  
- $L \leq e \leq U$

[[Floating Points|Read More]]

## Conversion
Consider the floating point of `13.625`

First, we convert the `whole number` to binary.
$$
\begin{align}
13 \div 2  = 6 \text{ R } 1 \\[6pt]
6 \div 2  = 3 \text{ R } 0 \\[6pt]
3 \div 2  = 1 \text{ R } 1 \\[6pt]
1 \div 2  = 0 \text{ R } 1 \\[6pt]
\end{align}
$$
Hence `13 -> 1101`. Count from `bottom-up`

Secondly, we convert the `fraction part` to binary.
$$
\begin{align}
0.625 \times 2 = 1.25 \to \text{Integer 1} \\[6pt]
0.25 \times 2 = 0.5 \to \text{Integer 0} \\[6pt]
0.5 \times 2 = 1 \to \text{Integer 1} \\[6pt]
\end{align}
$$
Hence `0.625 -> 101`. Count from `top-down`

Thus, `13.625 -> 1101.101`.
If we want to normalize it to `IEEE` form, `13.625 -> 1.101101 x 2^3`

## Edge Case
Consider the floating point `0.1`.

Try converting `0.1` to `binary`.
$$
\begin{align}
0.1 \times 2 = 0.2 \to 0 \\[6pt]
0.2 \times 2 = 0.4 \to 0 \\[6pt]
0.4 \times 2 = 0.8 \to 0 \\[6pt]
0.8 \times 2 = 1.6 \to 1 \\[6pt]
0.6 \times 2 = 1.2 \to 1 \\[6pt]
\dots \\
0.1 \times 2 = 0.2 \to 0 \\[6pt]
0.2 \times 2 = 0.4 \to 0 \\[6pt]
0.4 \times 2 = 0.8 \to 0 \\[6pt]
0.8 \times 2 = 1.6 \to 1 \\[6pt]
0.6 \times 2 = 1.2 \to 1 \\[6pt]
\end{align}
$$
It ends in repeating blocks of `0011`.

For such floating points, the we cut off after a fixed number of `mantissa`. (`23` in single precision, `52` in double precision)

## Binary Termination Condition
Only floating points whose `denominator` is a power of 2 can be accurately represented in binary.

- `0.1` $= \frac{1}{10}$ where $10$ cannot be represented as $2^n$.
- `0.75` $= \frac{3}{4}$ where $4$ can be represented as $2^2$.
