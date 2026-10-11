## Degrees of Freedom
[[Degrees of Freedom(DOF)|Degrees of freedom]] is the dimension of the [[Degrees of Freedom(DOF)|C-Space]].
![200](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSeNbwJFd76lA8pFFha1nJj8x8lsDjdJz_fL4v9SDrZbv1gYpNErFLru7iJ&s=10)

where the [[Degrees of Freedom(DOF)|C-space]] is the space of all configurations.

---
## General Formula 
The dimension of the [[Degrees of Freedom(DOF)|C-space]] is defined as
$$
\text{DOF} = \sum (\text{freedom of points}) - \# \text{ of independent constraints}
$$

Since robots are made made of [[Rigid Body|rigid bodies]], we can denote it as
$$
\text{DOF} = \sum (\text{freedom of bodies}) - \# \text{ of independent constraints}
$$

---
## Example with 2-DOF robot
Consider a [[Degrees of Freedom(DOF)|2-DOF robot]] as follow:
![350](https://automaticaddison.com/wp-content/uploads/2020/08/2dof-robotic-arm-sweep-gif.gif)

We can visualize angle of joint-1 as a point on a circle, and joint-2 as a point on anther circle:

![image|200](https://notes-media.kthiha.com/Degrees-of-Freedom/8a2d0608a3b84ee9180a736a78ec6726.png)

Rotate circle for joint-1 to be perpendicular to circle for joint-2:
![image|200](https://notes-media.kthiha.com/Degrees-of-Freedom/897834728e36d2021a996c26b8ffa26d.png)

At each angle of joint-1, there is a circle of possible joint angles for joint-2:
![image|200](https://notes-media.kthiha.com/Degrees-of-Freedom/28042bfcd4a7d72e3f3e4a8b2299cb97.png)

$\therefore$ The [[Degrees of Freedom(DOF)|c-space]] of a 2-joint robot can be visualized as a torus:

![200](https://upload.wikimedia.org/wikipedia/commons/8/8f/Ring_Torus_to_Degenerate_Torus_%28Short%29.gif?utm_source=en.wikipedia.org&utm_campaign=parser&utm_content=thumbnail_unscaled)

---
## DOF of Rigid Body
A [[Rigid Body|rigid body]] has a [[Degrees of Freedom(DOF)|degrees of freedom]] of $6$.

### Visualizing
First, we can choose an arbitrary point $A$ on the rigid body with $3$ degrees of freedom $(x,y,z)$.
![image|200](https://notes-media.kthiha.com/Degrees-of-Freedom/7259922b0d74c2df20407c2d19ff5a3b.png)

Then, if we choose another point $B$ on rigid body, it has to be on surface of sphere with fixed distance from $A$. Since we need only $2$ numbers to represent a point on a sphere, it has $2$ [[Degrees of Freedom(DOF)|DOF]].

![image|200](https://notes-media.kthiha.com/Degrees-of-Freedom/3a25418eb4d27dec7b4dcee57715553f.png)

Now, if we choose another point $C$, it has to be in intersection of spheres centered at $A$ and $B$. Hence, it has $1$ [[Degrees of Freedom(DOF)|DOF]].

---
### Constraints
![image|400](https://notes-media.kthiha.com/Degrees-of-Freedom(DOF)/1d46a584330588a8d2df14039d252dbe.png)

Hence, a [[Rigid Body|rigid body]] in $3D$ has $6$ degrees of freedom: 
- $3$ of which are linear: $x, \ y, \ z$
- $3$ of which are angled: $\text{roll, pitch, yaw}$

#### Rigid Body in 2D
We can use a similar process to get that a [[Rigid Body|rigid body]] in $2D$ has $3$ degrees of freedom:
- $2$ of which are linear
- $1$ of which are angled

#### Rigid Body in 3D
A [[Rigid Body|rigid body]] in $4D$ has $10$ degrees of freedom:
- $4$ of which are linear
- $6$ of which are angled

---
## See Also
- [Modern Robotics Video](https://modernrobotics.northwestern.edu/nu-gm-book-resource/2-1-degrees-of-freedom-of-a-rigid-body/#department)
- [[Degrees of Freedom(DOF)]]
- [[Rigid Body]]