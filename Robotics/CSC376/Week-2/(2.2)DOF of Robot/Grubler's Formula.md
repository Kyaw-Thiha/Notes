# Grubler's Formula
$$
\boxed{\quad \text{DOF} = m(N-1-J) + \sum^{J}_{i=1} f_{i} \quad }
$$

---
## Derivation

![image|300](https://notes-media.kthiha.com/Joint/cf66efb6efe5557bb973c6969f8a6124.png)

Recall that the [[Degrees of Freedom(DOF)|degrees of freedom]] can be calculated as
$$
\text{DOF} = \sum (\text{freedom of bodies}) - \# \text{ of independent constraints}
$$

Given the following:
- $N = \# \text{ of bodies, including ground}$
- $J = \# \text{ of joints}$
- $m = 6 \text{ for spatial bodies, } 3 \text{ for planar bodies}$

We can then derive [[Grubler's Formula|Grubler's formula]] as follows:
$$
\begin{align}
\text{DOF}
&= \underbrace{m(N-1)}_{\text{rigid body freedoms}}
- \underbrace{\sum_{i=1}^{J} c_{i}}_{\text{joint constraints}} \\[6pt]
&= \underbrace{m(N-1)}_{\text{rigid body freedoms}}
- \underbrace{\sum_{i=1}^{J} (m-f_{i})}_{\text{joint constraints}} \\[6pt]
&= m(N-1-J) + \sum^{J}_{i=1} f_{i}
\end{align}
$$

where constraints provided by the joints are independant.

---
## Serial(Open-Chain) 3R Robot
Its called a serial(open-chain) robot since there is a single path from ground to the end of the robot.

![200](https://media.cheggcdn.com/media/608/608b47c4-e278-4b51-ac67-fa427b0fd0b1/phpg1mUr6)

Its called a 3R robot meaning it has $3$ [[Joint|revolute joints]].

---
### Computing Degrees of Freedom
This robot has 
- $m=3$(since its planar)
- $N=4$($3$ bodies + ground)
- $J=3$(3 joints)

Using [[Grubler's Formula|Grubler's formula]], we get 
$$
3(4 - 1 - 3) + 3 = 3
$$
Hence, the robot has $3$ [[Degrees of Freedom(DOF)|degrees of freedom]].

---
## Four-Bar Linkage
[[#Four-Bar Linkage]] is obtained by pinning the endpoint of the [[#Serial(Open-Chain) 3R Robot|3R robot]] to a fixed position on the plane.

![300](https://www.mechref.org/dyn/four_bar_linkages_applications/4Bar.png)

This is called a closed-chain mechanism because there is a closed-loop.

---
### Computing Degrees of Freedom
This robot has 
- $m=3$(since its planar)
- $N=4$($3$ bodies + ground)
- $J=4$($4$ joints)

![300](https://europe1.discourse-cdn.com/unity/original/4X/c/5/9/c596d7c7bd2c30d8c8bb249ba5ce4a87efc0a8e2.gif)

Using [[Grubler's Formula|Grubler's Formula]], we get 
$$
3(4-1-4) + 4 = 1
$$
Hence, the robot has $1$ [[Degrees of Freedom(DOF)|degrees of freedom]].

> We can also predict by that pinning the endpoint of a 3R robot to a particular x-y location creates two constraints.

---
## Where Grubler's Formula does not work
This mechanism is like the [[#Four-Bar Linkage]] but now it adds $1$ more link and $2$ more [[Joint|joints]].

![image|300](https://notes-media.kthiha.com/Grubler's-Formula/e88750fbb20efd3b7e4069541b0bed95.png)

[[Grubler's Formula]] will tell us that this has $0$ degrees of freedom.
But this is wrong since it still have $1$ degree of freedom.

The reason for this is that the constraints are not independant.

---
## Stewart Platform
[[Stewart Platform]] is a spatial close-chained mechanism.

![300](https://cdn.prod.website-files.com/5fd37b46bba1787ca7d07db3/65d746236f8b5031ca2d0ca5_stewart_platform_banner%20(1).webp)

It has
- $6$ legs connecting bottom platform to the top platform
- Each leg consists of 
	- $2$ links
	- $1$ universal joint
	- $1$ prismatic joint
	- $1$ spherical joint

Since each leg has $2$ links and adding bottom and top platform,
$$
\text{links = } 2 \times 6 + 2 = 14
$$

Each leg has $3$ joints with $6$ degrees of freedom.
This total to $18$ joints with $36$ total freedoms.

The mechanism move in 3-dimensional space, so $m=6$.

Using [[Grubler's Formula]], we get
$$
6(14 - 1- 18) + 36 = 6
$$

Hence, it has $6$ [[Degrees of Freedom(DOF)|degrees of freedom]].

![300](https://i.imgur.com/3gayHWT.gif)

There are limits to range of motions, but these limits do not reduce the degrees of freedom.

---
## See Also
- [Modern Robotics Video](https://modernrobotics.northwestern.edu/nu-gm-book-resource/2-2-degrees-of-freedom-of-a-robot/)
- [[Joint]]
- [[Grubler's Formula]]
- [[Stewart Platform]]
- [[Degrees of Freedom(DOF)]]
- [[Rigid Body]]