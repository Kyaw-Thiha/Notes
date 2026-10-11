# Stewart Platform
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
