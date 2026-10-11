# DH Parameter
In the [[DH Parameter (Denavit-Hartenberg)|DH Parameter method]], we 
- attach reference frames to each link
- concatenate transformations from one link to the next

![image|500](https://notes-media.kthiha.com/DH-Parameter-(Denavit-Hartenberg)/f7913b2fbeb7a9a24c2a419b2409298c.png)

---
## DH Parameter
There are four parameters for each of the two adjacent links:

![image|500](https://notes-media.kthiha.com/DH-Parameter-(Denavit-Hartenberg)/7ac61903e84477fcc5931b0c62f48759.png)

- `Link Length`: $a_{i-1}$  
  Length of mutually perpendicular line between axis $i-1$ and axis $i$.
- `Link Twist`: $\alpha_{i-1}$
  Angle from $\hat{z}_{i-1}$ to $\hat{z}_{i}$ measured about $\hat{x}_{i-1}$.
- `Link Offset`: $d_{i}$
  Distance from intersection $\hat{x}_{i-1}$ to $\hat{z}_{i}$ to origin of link $i$ frame.
- `Joint Angle`: $\phi_{i}$
  Angle from $\hat{x}_{i-1}$ to $\hat{x}_{i}$ measured about $\hat{z}_{i}$

Note that the above photo is for [[Joint|revolute joints]].

---
### Prismatic Joint
The [[Joint|prismatic joint]] has the following [[DH Parameter (Denavit-Hartenberg)|DH paramters]]:

![image|500](https://notes-media.kthiha.com/DH-Parameter-(Denavit-Hartenberg)/6fb8b86ed7a82a21fa6f635721156fc6.png)

---
## Transformation between Adjacent Link Frames
Consider the following transformation between adjacent link frames:
$$
T_{i-1, \ i} = Rot(\hat{x}, \ \alpha_{i-1}) \ \text{Trans}(\hat{x}, \ 
\alpha_{i-1}) \ \text{Trans}(\hat{z}, \ d_{i}) \ Rot(\hat{z}, \ \phi_{i})
$$
![image|300](https://notes-media.kthiha.com/DH-Parameter-(Denavit-Hartenberg)/263e67b2b142be8eec8493c1ec42f0bd.png)

### Rotation $\alpha_{i-1}$ about $\hat{x}_{i-1}$.

$$
Rot(\hat{x}, \ \alpha_{i-1})
= \begin{bmatrix}
1 & 0 & 0 & 0 \\[6pt]
0 & \cos(\alpha_{i-1}) & -\sin(\alpha_{i-1}) & 0 \\[6pt]
0 & \sin(\alpha_{i-1}) & \cos(\alpha_{i-1}) & 0 \\[6pt]
0 & 0 & 0 & 1 \\[6pt]
\end{bmatrix}
$$
![image|400](https://notes-media.kthiha.com/DH-Parameter-(Denavit-Hartenberg)/0c0cc2f1f1210df45ec9df7850c5c3b9.png)

### Translation $a_{i-1}$ along $\hat{x}_{i-1}$.
$$
\text{Trans}(\hat{x}, \ a_{i-1})
= \begin{bmatrix}
1 & 0 & 0 & a_{i-1} \\[6pt]
0 & 1 & 0 & 0 \\[6pt]
0 & 0 & 1 & 0 \\[6pt]
0 & 0 & 0 & 1
\end{bmatrix}
$$
![image|400](https://notes-media.kthiha.com/DH-Parameter-(Denavit-Hartenberg)/517ccc6963b276b186997593cd86a2da.png)

### Translation of $d_{i}$ along $\hat{z}_{i}$.
$$
\text{Trans}(\hat{z}, \ d_{i})
= \begin{bmatrix}
1 & 0 & 0 & 0 \\[6pt]
0 & 1 & 0 & 0 \\[6pt]
0 & 0 & 1 & d_{i} \\[6pt]
0 & 0 & 0 & 1 \\[6pt]
\end{bmatrix}
$$
![image|400](https://notes-media.kthiha.com/DH-Parameter-(Denavit-Hartenberg)/3de21e85283fb729dd020756102c2565.png)

### Rotation of $\phi_{i}$ about $z_{i}$.
$$
Rot(\hat{z}, \phi_{i})
= \begin{bmatrix}
\cos(\phi_{i}) & -\sin(\phi_{i}) & 0 & 0 \\[6pt]
\sin(\phi_{i}) & \cos(\phi_{i}) & 0 & 0 \\[6pt]
0 & 0 & 1 & 0 \\[6pt]
0 & 0 & 0 & 1 \\[6pt]
\end{bmatrix}
$$
![image|400](https://notes-media.kthiha.com/DH-Parameter-(Denavit-Hartenberg)/7a8cf6d8694e9c224c65d8e1974e47a6.png)

### Final
Combining all of them, we get 
$$
T_{i-1, \ i}
= \begin{bmatrix}
\cos(\phi_{i}) & -\sin(\phi_{i}) & 0 & a_{i-1} \\[6pt]

\sin(\phi_{i})\cos(\alpha_{i-1}) & \cos(\phi_{i})\cos(\alpha_{i-1})
& -\sin(\alpha_{i-1}) & -d_{i}\sin(\alpha_{i-1}) \\[6pt]

\sin(\phi_{i})\sin(\alpha_{i-1}) & \cos(\phi_{i})\sin(\alpha_{i-1})
& \cos(\alpha_{i-1}) & d_{i}\cos(\alpha_{i-1}) \\[6pt]

0 & 0 & 0 & 1
\end{bmatrix}
$$

---
## Transformation Concatentation
The [[Forward Kinematics|forward kinematics]] is obtained by sequentially multiplying the link transformations:
$$
T_{s, \ b}(\theta) = T_{s, \ 1}\left( \prod^{n}_{i=2} T_{i-1, \ i} \right)
T_{n, \ b}
$$
as defined in

![image|300](https://notes-media.kthiha.com/DH-Parameter-(Denavit-Hartenberg)/f7913b2fbeb7a9a24c2a419b2409298c.png)

---
## Assign Link Frame
- Identify the joint axes and imagine infinite lines along them.
- $\hat{z}_{i}$ coincides with joint axis $i$ and the $\hat{z}_{i-1}$-axis coincides with joint axis $i-1$.
- Place origin of link frame $i-1$ where mutually perpendicular line connecting $\hat{z}_{i-1}$ and $\hat{z}_{i}$ intersects $\hat{z}_{i-1}$
- $\hat{x}_{i-1}$ in direction of mutually perpendicular line pointing from joint axis $i-1$ to axis $i$
- $\hat{y}_{i}$ uniquely determined from cross-product $\hat{x} \times \hat{z}_{i}$

---
### Special Cases
> **Adjacent Joint Axes Intersect**
> Choose $\hat{x}_{i-1}$ perpendicular to plane spanned by $\hat{z}_{i-1}$ and $\hat{z}_{i}$.

> **Adjacent Parallel Joint Axes**
> Choose mutually perpendicular line that is most physically intuitive and results in as many zero [[DH Parameter (Denavit-Hartenberg)|DH parameters]] as possible.

---
