# Week-5 Tutorial
## Question
The following is a RPR serial kinematic chain where the end-effector is expressed as a frame $\{ b \}$. The robot is depicted in its zero configuration $(\text{all }\theta=0)$.

![image|400](https://notes-media.kthiha.com/CSC376-Week-5-Tutorial/84e369f6b4b84d44bd7196d87ed80c81.png)

Derive the velocity kinematics (geometric Jacobian) of the manipulator depicted for the different scenarios.

---
### Q1
The end-effector velocity is expressed as a twist $\mathcal{V}_{s}$ in the fixed frame $\{ s \}$ (Space Jacobian) and all $\theta=0$.

### Solution
The velocity kinematics of the spatial twist can be described as 
$$
\mathcal{V}_{s} = J_{s}(\theta) \ \dot{\theta}
$$
where
- $J_{s}(\theta) = [ \ J_{s1} \ J_{s2}(\theta) \ \dots \ J_{sn}(\theta) \ ] \in \mathbb{R}^{6\times n}$
	- with $J_{si}(\theta) = \text{Ad}_{e^{[S_{1}] \ \theta_{1}} \ \dots \ e^{[S_{i-1}] \ \theta_{i-1}} (S_{i})}$
- $\dot{\theta} \in \mathbb{R}^{n}$

#### Determining $S_{i}$

| $S_{i}$ | $\omega_{i}$ | $q_{i}$     | $v_{i}$     |
| ------- | ------------ | ----------- | ----------- |
| 1       | $(0, 1, 0)$  | $(0,0,0)$   | $(0, 0, 0)$ |
| 2       | $(0, 0, 0)$  | $(0,0,0)$   | $(0,1,0)$   |
| 3       | $(0, 0, 1)$  | $(0,2L, 0)$ | $(-2L,0,0)$ |

#### Computing Columns of $J_{s}(\theta)$
This gets us the Jacobian matrix of
$$
J_{s}(\theta) = \begin{bmatrix}
0 & 0 & 0 \\[6pt]
1 & 0 & 0 \\[6pt]
0 & 0 & 1 \\[6pt]
0 & 0 & -2L \\[6pt]
0 & 1 & 0 \\[6pt]
0 & 0 & 0 \\[6pt]
\end{bmatrix}
$$
This gets us the spatial twist of
$$
\mathcal{v}_{s} = J_{s}(\theta) \ \dot{\theta}
= \begin{bmatrix}
0 \\[6pt]
\dot{\theta}_{1} \\[6pt]
\dot{\theta}_{3} \\[6pt]
-2L \cdot \dot{\theta}_{3} \\[6pt]
\dot{\theta}_{2} \\[6pt]
0 \\[6pt]
\end{bmatrix}
$$

---
### Q2
The end-effector velocity is expressed as a twist $\mathcal{V}_{s}$ in the body frame $\{ b \}$ (Body Jacobian) and all $\theta=0$.

### Solution
The velocity kinematics of the spatial twist can be described as 
$$
\mathcal{V}_{s} = J_{s}(\theta) \ \dot{\theta}
$$
where
- $J_{s}(\theta) = [ \ J_{s1} \ J_{s2}(\theta) \ \dots \ J_{sn}(\theta) \ ] \in \mathbb{R}^{6\times n}$
	- with $J_{si}(\theta) = \text{Ad}_{e^{[S_{1}] \ \theta_{1}} \ \dots \ e^{[S_{i-1}] \ \theta_{i-1}} (S_{i})}$
- $\dot{\theta} \in \mathbb{R}^{n}$

#### Determining $S_{i}$

| $S_{i}$ | $\omega_{i}$ | $q_{i}$     | $v_{i}$     |
| ------- | ------------ | ----------- | ----------- |
| 1       | $(0, 1, 0)$  | $(0,0,0)$   | $(0, 0, 0)$ |
| 2       | $(0, 0, 0)$  | $(0,0,0)$   | $(0,1,0)$   |
| 3       | $(0, 0, 1)$  | $(0, L, 0)$ | $(-L,0,0)$  |

#### Computing Columns of $J_{s}(\theta)$
This gets us the Jacobian matrix of
$$
J_{s}(\theta) = \begin{bmatrix}
0 & 0 & 0 \\[6pt]
1 & 0 & 0 \\[6pt]
0 & 0 & 1 \\[6pt]
0 & 0 & -2L \\[6pt]
0 & 1 & 0 \\[6pt]
0 & 0 & 0 \\[6pt]
\end{bmatrix}
$$
