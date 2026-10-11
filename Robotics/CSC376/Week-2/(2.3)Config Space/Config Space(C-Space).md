# Config Space
The [[Config Space(C-Space)|C-space]] is the set of all possible positions, angles and values that a [[Joint|robot's joint]] or end-effector can take.

![300](https://stevengong.co/attachments/Pasted-image-20211124110732.png)

[[Degrees of Freedom(DOF)|Degrees of freedom]] is the dimension of the [[Degrees of Freedom(DOF)|C-Space]].

---
## Topology
The [[Config Space(C-Space)|C-Space]] can have a distinct [[Topology|topology]]. 
For example, consider a [[Degrees of Freedom(DOF)|2-DOF robot]] as follow:

![350](https://automaticaddison.com/wp-content/uploads/2020/08/2dof-robotic-arm-sweep-gif.gif)

Its [[Topology|topology]] can be represented as a [[Topology|torus]]:

![200](https://upload.wikimedia.org/wikipedia/commons/8/8f/Ring_Torus_to_Degenerate_Torus_%28Short%29.gif?utm_source=en.wikipedia.org&utm_campaign=parser&utm_content=thumbnail_unscaled)

---
## Config Space Representation
A [[Topology|topology]] can be represented in either [[#Explicit Parameterization|explicit parameterization]] and [[#Implicit Representation|implicit representation]].

![image|300](https://notes-media.kthiha.com/Config-Space(C-Space)/3d360b6b2528c250a45e654ae356dad7.png)

---
### Explicit Parameterization
[[#Explicit Parameterization|Explicit parameterization]] uses a minimum number of coordinates to represent the [[Config Space(C-Space)|C-Space]].

![image|300](https://notes-media.kthiha.com/Config-Space(C-Space)/2444599b367ddab16ff01fb617391836.png)

- **Advantage**: Simplicity of a minimum number of coordinates 
- **Disadvantage**: Because the topology of the space is different from a Euclidean space, the representation will have poor behavior at some points of the space.

---
### Implicit Representation
An [[#Implicit Representation|implicit representation]] views the n-dimensional space as embedded in a higher-dimensional Euclidean space.

![image|300](https://notes-media.kthiha.com/Config-Space(C-Space)/d06090b6756e23ec714838f4734f2bd1.png)

- **Advantage**: Using x-y-z coordinates subject to one constraint, there are no problems anywhere with discontinuities or rapidly changing coordinates.
- **Disadvantage**: Somewhat greater complexity of the representation.

---
## See Also
- [Modern Robotics Video on C-Space Topology](https://modernrobotics.northwestern.edu/nu-gm-book-resource/2-3-1-configuration-space-topology/)
- [Modern Robotics Video on C-Space Representation](https://modernrobotics.northwestern.edu/nu-gm-book-resource/2-3-2-configuration-space-representation/)
- [[Degrees of Freedom(DOF)]]
- [[Topology]]