# Special Orthogonal group SO(3)
[[SO(3) Rotation|SO(3)]] is the special orthogonal group that represents rotations.

![image|500](https://notes-media.kthiha.com/SO(3)-Rotation/0de40987ca253cba03e8518d7688b1f1.png)

where
$$
p_{s} = \begin{bmatrix}
p_{x} \\ p_{y} \\ p_{z}
\end{bmatrix} \in \mathbb{R}^{3}
\ , \quad
R = \begin{bmatrix}
\hat{x}_{b} & \hat{y}_{b} & \hat{z}_{b}
\end{bmatrix}
= \begin{bmatrix}
r_{11} & r_{12} & r_{13} \\
r_{21} & r_{22} & r_{23} \\
r_{31} & r_{32} & r_{33} \\
\end{bmatrix} \in \mathbb{R}^{3\times3}
$$

---
## Formal Definition
The [[SO(3) Rotation|special orthogonal group]] $SO(3)$, also known as group of rotation matrices, is the set of all $3\times3$ matrices $R$ that satisfy
- $R^{T}R = I$
- $\det R = 1$

---
## Properties
![image|500](https://notes-media.kthiha.com/SO(3)-Rotation/44bf714501a719abec56487ab4137e58.png)

---
## Common Rotation Matrices
![image|500](https://notes-media.kthiha.com/SO(3)-Rotation/cfdb3aeaa4e962e2292970e53bb3a4a5.png)

---
## Euler's Rotation Theorem
> `Theorem`
> Any rotation can be achieved from an initial orientation $I$ by rotating about some axis $\hat{\omega}$ with particular angle $\theta$.

![image|150](https://notes-media.kthiha.com/SO(3)-Rotation/ef1386b016bfa6559ac8928d8aeaec22.png)

Let $\hat{w} = \begin{bmatrix}\hat{w}_{1} & \hat{w}_{2} & \hat{w}_{3} \end{bmatrix}$, $c_{\theta} = \cos(\theta)$ and $s_{\theta} = \sin(\theta)$. Then, we have

$$
R(\hat{w},\theta)
=
\begin{bmatrix}
c_{\theta} + \hat{w}_{1}^{2}(1-c_{\theta})
&
\hat{w}_{1}\hat{w}_{2}(1-c_{\theta})-\hat{w}_{3}s_{\theta}
&
\hat{w}_{1}\hat{w}_{3}(1-c_{\theta})+\hat{w}_{2}s_{\theta}
\\[6pt]
\hat{w}_{1}\hat{w}_{2}(1-c_{\theta})+\hat{w}_{3}s_{\theta}
&
c_{\theta}+\hat{w}_{2}^{2}(1-c_{\theta})
&
\hat{w}_{2}\hat{w}_{3}(1-c_{\theta})-\hat{w}_{1}s_{\theta}
\\[6pt]
\hat{w}_{1}\hat{w}_{3}(1-c_{\theta})-\hat{w}_{2}s_{\theta}
&
\hat{w}_{2}\hat{w}_{3}(1-c_{\theta})+\hat{w}_{1}s_{\theta}
&
c_{\theta}+\hat{w}_{3}^{2}(1-c_{\theta})
\end{bmatrix}
$$

Note that $Rot(\hat{w}, \theta) = Rot(-\hat{w}, - \theta)$.

---
