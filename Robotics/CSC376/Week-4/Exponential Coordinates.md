# Exponential Coordinates of Rigid Body Motion
## Chasles-Mozzi Theorem
> Every [[Rigid Body|rigid-body displacement]] can be expressed as a displacement along a [[Screw Axis|fixed screw axis]] $\mathcal{S}$ in space.
> 
> $S\theta \in \mathbb{R}^{6}$ are the [[Exponential Coordinates|exponential coordinates]] of a homogeneous transformation matrix $T \in SE(3)$, where $\mathcal{S}$ is the [[Screw Axis|screw axis]] and $\theta$ scalar parameter describing the magnitude of motion along the screw axis to take a frame from the origin $I$ to $T$.

Note that
- for a [[Joint|revolute joint]], $\theta$ is the rotation angle
- for a [[Joint|prismatic joint]], $\theta$ is the translation angle

---
## Screw Axis and Exponential Coordinates
Let $S = \begin{bmatrix} \omega \\ v \end{bmatrix} \in \mathbb{R}^{6}$ be the [[Screw Axis|screw axis]].
Applying the [[Exponential Coordinates|exponential coordinates]], we get 
$$
S\theta = \begin{bmatrix}
\omega \\ v
\end{bmatrix} \theta \in \mathbb{R}^{6}
$$
To represent this in $4\times4$ matrix representation of a [[Screw Axis|screw axis]], we have
$$
[S] = \begin{bmatrix}
[w] & v \\[6pt]
0 & 0
\end{bmatrix} \in se(3)
$$
where $se(3)$ is the Lie Algebra of [[SE(3) Transformation|SE(3)]], and
$$
[w] = \begin{bmatrix}
0 & - \omega_{3} & \omega_{2} \\[6pt]
\omega_{3} & 0 & -\omega_{1} \\[6pt]
-\omega_{2} & \omega_{1} & 0 \\[6pt]
\end{bmatrix} \in so(3)
$$
where $so(3)$ is the Lie Algebra of [[SO(3) Rotation|SO(3)]].

---
## Matrix Exponential
Mapping from $se(3)$ to $SE(3)$, we have
$$
\exp: [\mathcal{S}] \theta \in se(3) \to T \in SE(3)
$$
Using Taylor Series of $\exp x:= \sum^{\infty}_{k=0} \frac{x^{k}}{k!},$ we have
$$
\begin{align}
e^{[S]\theta}
&= I + [S] \theta + [S]^{2} \frac{\theta^{2}}{2!}  
+ [S]^{3} \frac{\theta^{3}}{3!} + \dots \\[6pt]

&= \begin{bmatrix}
R & \mathbf{p} \\
0 & 1
\end{bmatrix} \\[6pt]

&= 
\begin{bmatrix}
I + \sin\theta\,[\boldsymbol{\omega}] + (1-\cos\theta)[\boldsymbol{\omega}]^2
&
\left(I\theta + (1-\cos\theta)[\boldsymbol{\omega}] + (\theta-\sin\theta)[\boldsymbol{\omega}]^2\right)\mathbf{v} \\
0 & 1
\end{bmatrix}
\end{align}
$$

### Rotation
This means that for $||\omega||=1$, we get
$$
e^{[\mathcal{S}]\theta} =
\begin{bmatrix}
e^{[\boldsymbol{\omega}]\theta} & \left(I\theta + (1-\cos\theta)[\boldsymbol{\omega}] + (\theta-\sin\theta)[\boldsymbol{\omega}]^2\right)\mathbf{v} \\
0 & 1
\end{bmatrix}
$$
This actually get us the `Rodrigues` formula for rotations:
$$
Rot(\omega, \theta) = 
e^{[\boldsymbol{\omega}]\theta} = I + \sin\theta\,[\boldsymbol{\omega}] + (1-\cos\theta)[\boldsymbol{\omega}]^2 \;\in\; SO(3)
$$

### Translation
For $\omega=0$ and $||v||=1$, we get
$$
e^{[\mathcal{S}]\theta} =
\begin{bmatrix}
I & \mathbf{v}\theta \\
0 & 1
\end{bmatrix}
\in SE(3)
$$

---
## Example
> Determine $T_{sa'}$ using an [[Exponential Coordinates|exponential coordinate representation]].

![image|300](https://notes-media.kthiha.com/Exponential-Coordinates/8dbdef04866ea506f5fb6ecf0ea24885.png)

> **Step-1**: Determine $S = \begin{bmatrix} \omega \\ v \end{bmatrix} \in \mathbb{R}^{6}$.

$$
\omega = \hat{z} = \begin{bmatrix}
0 \\ 0 \\ 1
\end{bmatrix}
\quad \quad \text{ and } \quad \quad
v = - \omega \times q 
= - \begin{bmatrix}
0 \\ 0 \\ 1
\end{bmatrix} 
\times \begin{bmatrix}
0 \\ 3 \\ 0
\end{bmatrix}
= \begin{bmatrix}
3 \\ 0 \\ 0 
\end{bmatrix}
$$
> **Step-2**: Use [[Exponential Coordinates|matrix exponential]] to determine $T_{sa'}$.

Using $e^{[\boldsymbol{\omega}]\theta} = I + \sin\theta\,[\boldsymbol{\omega}] + (1-\cos\theta)\,[\boldsymbol{\omega}]^2 \in SO(3)$, we compute the rotation component 
$$
e^{[\boldsymbol{\omega}]\theta}
= \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}
+ \sin\pi[\omega] 
+ (1 - \cos \theta)\begin{bmatrix} 0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix}^{2}
= \underbrace{\begin{bmatrix}
-1 & 0 & 0 \\[6pt]
0 & -1 & 0 \\[6pt]
0 & 0 & 1 \\[6pt]
\end{bmatrix}}_{Rot(\hat{z}, \pi)}
$$
Using $\left(I\theta + (1-\cos\theta)[\boldsymbol{\omega}] + (\theta-\sin\theta)[\boldsymbol{\omega}]^2\right)\mathbf{v}$, we compute the translational component 
$$
\left(
\begin{bmatrix} \pi & 0 & 0 \\ 0 & \pi & 0 \\ 0 & 0 & \pi \end{bmatrix}
+ 2\begin{bmatrix} 0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix}
+ \pi\begin{bmatrix} 0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix}^2
\right)
\begin{bmatrix} 3 \\ 0 \\ 0 \end{bmatrix}
=
\begin{bmatrix} 0 \\ 6 \\ 0 \end{bmatrix}
$$
This get us the transformation of 
$$
T_{sa'} = e^{[\mathcal{S}]\theta} =
\begin{bmatrix}
-1 & 0 & 0 & 0 \\
0 & -1 & 0 & 6 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

---