# Week-4
## Exponentials Cheatsheet
The rotational matrix is
$$
[\mathbf{a}] = \begin{bmatrix} 0 & -a_3 & a_2 \\ a_3 & 0 & -a_1 \\ -a_2 & a_1 & 0 \end{bmatrix} \qquad \begin{aligned} \cos\pi &= -1 \\ \sin\pi &= 0 \end{aligned} 
$$
We have the [[Exponential Coordinates|matrix exponential]] of 
$$
e^{[\mathcal{S}]\theta} = \begin{bmatrix} e^{[\omega]\theta} & \left(I\theta + (1-\cos\theta)[\omega] + (\theta-\sin\theta)[\omega]^2\right)\mathbf{v} \\ 0 & 1 \end{bmatrix}
$$
where $e^{[w]\theta}$ is 
$$
e^{[\omega]\theta} = I + \sin\theta\,[\omega] + (1-\cos\theta)\,[\omega]^2 \in SO(3)
$$

---