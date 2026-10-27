# Screw Axis
Every [[Rigid Body|rigid-body displacement]] can be expressed as a displacement along a fixed [[Screw Axis|screw axis]] $\mathcal{S}$ in space.

A [[Screw Axis|screw motion]] has two components
- [[#Angular Component|rotation(angular)]] about the screw axis
- [[#Linear Component|translational(linear)]] about the screw axis

---
## Screw Axes for Common Robot Joint Types
### Prismatic Joint
![image|200](https://notes-media.kthiha.com/Screw-Axis/f4d2f58388aab21598653248606e2521.png)

### Revolute Joint
![image|200](https://notes-media.kthiha.com/Screw-Axis/f1883f0701da131f3ad5d3f716e31985.png)

### Helical Joint
![image|200](https://notes-media.kthiha.com/Screw-Axis/65154ada02702a367edb91c68bf1b6af.png)

---
## Formal Definition
For a given [[Robotics Frame|reference frame]], a [[Screw Axis|screw axis]] $\mathcal{S}$ is written as $S=\begin{bmatrix}w & v\end{bmatrix} \in \mathbb{R}^{6}$, where either
- $(i)$ $||\omega|| = 1$
- $(ii)$ $\omega = 0$ and $||v||=1$

---
## Angular Component
The [[#Angular Component|angular component]] $\omega$ is
![image|300](https://notes-media.kthiha.com/Screw-Axis/14231042bb43fdc0dd59fd6a123c8e7c.png)

where $\theta$ is the angle along the $\omega$ axis.

---
## Linear Component
The [[#Linear Component|linear component]] $v$ can is

![image|300](https://notes-media.kthiha.com/Screw-Axis/42d5bb57a5e86a6c80fac423b6379e4c.png)

with the two components of

![image|200](https://notes-media.kthiha.com/Screw-Axis/ab4ad2ba8f0c7df14fa01893524f32e7.png)

and

![image|200](https://notes-media.kthiha.com/Screw-Axis/944ee387ba275980f254ccc3e0f0c71c.png)

---
