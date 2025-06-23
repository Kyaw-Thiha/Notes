#math #cv/transformations/3d/rotation/quarternion
# Quarternions in Rotation

Quarternions are 4D vectors $(a, b, c, d)$, that comprises of the scalar component $a$ and vector/imaginary component $\vec{u} = (b, c, d)$.
Compared to [[Davenport Rotation | Euler Rotations]], quarternions have several benefits including
- Not suffering from [[Gimbal Lock]]
- Lower storage space compared to matrices of Euler Rotations(4 parameters instead of 9 parameters)
- Better for continuous rotation compared to [[Axis-Angle Representation]]
- Better numerical stability
Their only weakness is that they can be hard to understand

## Defnition of Quarternion
Quarternion are essentially 4D vectors made up of a real/scalar value, and 3 imaginary/vector values.

Hence, they can be represented as
$$
\vec{q} = a + b.\hat{i} = c.\hat{j} + d.\hat{k}
$$
or
$$
\vec{q} = a + \vec{u}
$$
> IMPORTANT: note that in computer vision context, and scipy package, quarternion is denoted as (x, y, z, w), where w is the real component

Here are the main properties of quarternions.

- **Multiplicative table of Quarternion**
$$
	i^2 = j^2 = k^2 = -1
$$
$$
\begin{aligned}
ij &= -ji = k \\
jk &= -kj = i \\
ki &= -ik = j
\end{aligned}
$$
- **Vector Multiplication**: $\vec{p}.\vec{q}$
$$
(a+\vec{u})⋅(b+\vec{v})
=(ab−⟨\vec{u},\vec{v}⟩) + (av+bu+\vec{u}×\vec{v})
$$
- **Non-Commutativity**: $\vec{p}.\vec{q} \neq \vec{q}.\vec{p}$ 
- **Associativity**: $\vec{p}.(\vec{q}.\vec{r}) = (\vec{p}.\vec{q}).\vec{r}$
- **Length**: $|\vec{q}| = \sqrt{a^2 +b^2 + c^2 + d^2}$

## Rotation With Quarternion
The rotation formula to rotate any vector in 3D space using a unit quarternion is
$$
\vec{q}.\vec{v}.\vec{q^c}
$$
where 
- $\vec{v}$ is the 3D vector to be rotated
- $\vec{q}$ is the unit rotation quarternion as determined using the formula below

$$
\vec{q} = \cos\left( \frac{\theta}{2} \right) + 
\sin\left( \frac{\theta}{2} \right).\vec{n}
$$
where 
- $\vec{n}$ is the axis of rotation, and 
- $\theta$ is the angle of rotation about that axis.

## Relation to Rodrigues' Formula
Recall that [[Axis-Angle Representation#Rodrigues Formula |Rodrigues Formula]] is 
$$

$$


## Math behind Quarternions
If you want to learn more about all the properties, and their proofs, as well as the proof of [[Quarternions Math#Proving Quarternion's Relation to Rotation | why quarternion can be used to represent 3D rotations]] , you can read [[Quarternions Math |here]]


