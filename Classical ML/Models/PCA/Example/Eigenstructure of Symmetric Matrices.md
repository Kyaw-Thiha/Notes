# Eigenstructure of Symmetric Matrices
#math 

`Lemma`

Let $v \in R^n$ be a nonzero vector.
Define 
$$
S = vv^T \in R^{\ n \times n}
$$

Then,

1. $v$ is an `eigenvector` of $S$ with `eigenvalue`
$$
\lambda_{1} = v^Tv = ||v||^2
$$

2. Every vector $w$ `orthogonal` to $v$ $(\text{denoted} \ v^Tw = 0)$  is an `eigenvector` with `eigenvalue`:
$$
\lambda_{2} = 0
$$
3. Consequently, the `spectrum` of $S$ is 
$$
\{ ||v||^2,\ 0 ,\ 0,\ \dots,\ 0 \}
$$
with one `non-zero eigenvalue` $||v||^2$, and $(n-1)$ zeroes

---
`Proof`

Let $x \in R^n$ be arbitrary.
Then,
$$
Sx = (vv^T) \ x = v(v^Tx)
$$

This means that for any vector $x$, the output $Sx$ is always some `scalar multiple` of $v$.

`Case-1:`$x = v$
Hence, we get
$$
Sv = v(v^T v) = (v^Tv)\ v
$$
Thus,
$$
Sv = \lambda_{1}v 
$$
where $\lambda_{1} = v^Tv = ||v||^2$ 

`Case-2:` $x=w$, $v^Tw = 0$
Since $w$ is `orthogonal` to $v$, 
$$
Sw = v \ (v^Tw) = v \cdot 0 = 0
$$

Since $Sw = 0 \cdot w$, $w$ is an `eigenvector` of $S$ with `eigenvalue` $0$.

All such $w$ form the subspace
$$
v^{\ \perp} = \{ w \in R^n: v^Tw = 0 \}
$$
which has dimension $(n-1)$.

---
## See Also
- [[Rotating Coordinate System]]
