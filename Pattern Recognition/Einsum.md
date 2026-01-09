# Einsum
#math/linear-algebra/einsum 
`Einsum` is a technique to express tensor operations using index rules.

![Einsum|500](https://miro.medium.com/v2/resize:fit:1400/1*Kfonvq2Mm0Y4ql4pNlz6VA.png)

---
`Visual Example (Matrix Multiplication)`
Consider the following matrix multiplication.
$$
\begin{bmatrix}
A_{00} & A_{01} & A_{02} \\
A_{10} & A_{11} & A_{12} \\
A_{20} & A_{21} & A_{22} \\
A_{30} & A_{31} & A_{32} \\
\end{bmatrix}
\begin{bmatrix}
B_{00} & B_{01} \\
B_{10} & B_{11} \\
B_{20} & B_{21}
\end{bmatrix}
= \begin{bmatrix}
C_{00} & C_{01} \\
C_{10} & C_{11} \\
C_{20} & C_{21} \\
C_{30} & C_{31} \\
\end{bmatrix}
$$

Note that 
$$
\begin{align}
C_{00}  
&= A_{00} B_{00} + A_{01} B_{10} + A_{02} B_{20}  
\\[6pt]

C_{01}  
&= A_{00} B_{01} + A_{01} B_{11} + A_{02} B_{21}  
\\[6pt]

C_{10}  
&= A_{10} B_{00} + A_{11} B_{10} + A_{12} B_{20}  
\\[6pt]
\end{align}
$$

Hence, we can generalize it to get 
$$
C_{ij} = \sum_{k=0}^N A_{ik} B_{kj}
$$
where $N=2$

The compact rule can be used to represent it as
$$
\sum_{k} A_{ik}B_{kj} = C_{ij}
\quad \implies \quad
\boxed{\ ik, \ kj \to ij \ }
$$

---
## Rules
$$
\underbrace{ik}_{\text{Input-1}}, 
\ \underbrace{kj}_{\text{Input-2}}
\to \underbrace{ij}_{\text{Output}}
$$
1. If an `index label` in the output, the dimension is `kept`.
   $(\text{In this case, it is } i \text{ and } j)$
2. If an `index label` is not in the output, then that dimension is `removed` using whatever operation makes sense.
   $(\text{In this case, it is } k)$
3. If two dimensions both receive the `same index label`, then they must have the `same size`.
	- $\text{Suppose matrix A is } m \times n$.
	- $\text{Then, } \boxed{ii \to} \text{ means first } i \text{ represents axis 0 (size n) }$
     $\text{and second } i \text{ represents axis 1 (size m)}$
     - $\text{Since both are } i, \text{ n=m}$
4. The `order` of the output labels specifies the output's `final permutation`.
	- $ij \to ij$: keeps the same order $(n \times m)$
	- $ij \to ji$: swap labels $(m \times n)$

---
### Common Operations

`Summation Along Axis`
$$
\sum_{k} A_{ik} = C_{i}
\quad \implies \quad
ik \to i
$$

`Matrix Multiplication`
$$
\sum_{k} A_{ik} B_{kj} = C_{ij}
\quad \implies \quad
ik, \ kj \to ij
$$

`Tensor Contraction`
$$
\sum_{k} A_{ijk} B_{jil} = C_{kl} 
\quad \implies \quad
ijk, \ jil \to kl
$$

`Trace`
$$
\sum_{k} A_{kk} = C
\quad \implies \quad
kk \to
$$

`Permutation`
$$
A_{ji} = C_{ij}
\quad \implies \quad
ji \to ij
$$

`Elementwise Multiplication`
$$
A_{ij}B_{ij} = C_{ij}
\quad \implies \quad
ij, \ ij \to ij
$$

---
`Complex Example`
Consider the quadratic cost function
$$
X^TQX = C
$$
where
$$
Q = \begin{bmatrix}
Q_{00} & Q_{01} \\
Q_{10} & Q_{11}
\end{bmatrix}
\ , \quad
X = \begin{bmatrix}
X_{00} & X_{01} & X_{02} \\
X_{10} & X_{11} & X_{12} \\
\end{bmatrix}
$$

---
## See Also
- [Youtube Video on Einsum](https://youtu.be/IvgV6QcsC64?si=yzsJHmXAaWwIf7cM)
