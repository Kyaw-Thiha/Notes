# Special Euclidean Group SE(3)
The [[SE(3) Transformation|special Euclidean group SE(3)]], also known as group of rigid motions or homogeneous transformation matrices, is the set of all $4\times4$ real matrices $T$ of the form:
$$
T = \begin{bmatrix}
R & p \\[6pt]
0 & 1
\end{bmatrix}
= \begin{bmatrix}
r_{11} & r_{12} & r_{13} & p_{1} \\[6pt]
r_{21} & r_{22} & r_{23} & p_{2} \\[6pt]
r_{31} & r_{32} & r_{33} & p_{3} \\[6pt]
0 & 0 & 0 & 1
\end{bmatrix}
$$
where $R \in SO(3)$ and $p \in \mathbb{R}^{3}$ is the column vector. 

---
## Inverse
The [[#Inverse|inverse]] of a [[SE(3) Transformation|SE(3)]] matrix is defined as
$$
T^{-1} = \begin{bmatrix}
R & p \\[6pt]
0 & 1
\end{bmatrix}
= \begin{bmatrix}
R^{T} & -R^{T}p \\[6pt]
0 & 1
\end{bmatrix}
$$

---
## Use Cases of Transformation Matrices
### Configuration of Rigid Body
![image|400](https://notes-media.kthiha.com/SE(3)-Transformation/dd31c062e5a2afa427b18511c9cbcd07.png)

---
### Change of Reference Frame
![image|400](https://notes-media.kthiha.com/SE(3)-Transformation/20ebafef1e3c83ee1970ee741df27e22.png)

![image|400](https://notes-media.kthiha.com/SE(3)-Transformation/994f5c1a14d5d3b90bc0eebd433bcf1e.png)

---
### Display Vector/Frame
![image|400](https://notes-media.kthiha.com/SE(3)-Transformation/78dcd30ba4248598e8d4b3cdfe209fe8.png)

![image|400](https://notes-media.kthiha.com/SE(3)-Transformation/2e98264ff9f181ffd4de65a81b6fe05b.png)

![image|400](https://notes-media.kthiha.com/SE(3)-Transformation/1ce9f17c3713f23ba21f0c1484ae932d.png)

---
