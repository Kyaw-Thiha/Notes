# DH Examples
## Example-1
Consider the following `3R Planar Robot`.

![image|300](https://notes-media.kthiha.com/DH-Parameter-(Denavit-Hartenberg)/111e605ea991eafc1c933789891dbf38.png)

---
## Solution
### Zero-Config
First assign the [[Robotics Frame|frames]] to each of the [[Joint|joints]]. And start from the zero-configuration. 

![image|300](https://notes-media.kthiha.com/DH-Parameter-(Denavit-Hartenberg)/51a0437853ef863f4e16b2c21f3799e9.png)

### First Move
We first move the joints $1$ and $2$.

![image|300](https://notes-media.kthiha.com/DH-Parameter-(Denavit-Hartenberg)/7afac9d2bd21f034bc04fb22f3c38b1e.png)

This gets us the following transformations.

![image|300](https://notes-media.kthiha.com/DH-Parameter-(Denavit-Hartenberg)/8eaf26ffdc4cb4151c3d48216788a47a.png)

### Second Move
We then move the joint-$3$ to get us the final position. 
![image|300](https://notes-media.kthiha.com/DH-Parameter-(Denavit-Hartenberg)/111e605ea991eafc1c933789891dbf38.png)

This results in the following transformations of
![image|300](https://notes-media.kthiha.com/DH-Parameter-(Denavit-Hartenberg)/636102da769729977d576ea2b4c03823.png)

### Final Solution
Using this rule:
![image|150](https://notes-media.kthiha.com/DH-Parameter-(Denavit-Hartenberg)/2c8d47b4c92ea29680d3d49bcd2dd3b4.png)

we can get the following [[DH Parameter (Denavit-Hartenberg)|DH Table]] of
![image|300](https://notes-media.kthiha.com/DH-Parameter-(Denavit-Hartenberg)/fdf6280ed3d470a62e980dcc1a0f5cf0.png)

And the final transformation of
$$
T_{s,\ b} = T_{s, 1} \ T_{1,2} \ T_{2,3} \ T_{3,b}
= \begin{bmatrix}
1 & 0 & 0 & L_{1} + L_{2} + L_{3} \\[6pt]
0 & 1 & 0 & 0 \\[6pt]
0 & 0 & 1 & 0 \\[6pt]
0 & 0 & 0 & 1 \\[6pt]
\end{bmatrix}
$$

---
## Example-2
Consider the following `3R Spatial Robot`.

![image|300](https://notes-media.kthiha.com/DH-Examples/9698979bcd987a688f236f93a25c0a1e.png)

This robot is shown in its zero positions, with link frames pre-assigned.

### Solution
![image|300](https://notes-media.kthiha.com/DH-Examples/6a2712fc443e8c05014842c1aacb179a.png)

---
## Example-3
Consider the following `3R Spatial Robot`. 

![image|300](https://notes-media.kthiha.com/DH-Examples/f044b293031068c6997bad2e8cf0f047.png)

We will be [[DH Parameter (Denavit-Hartenberg)#Assign Link Frame|assigning link frames]] to it.

---
### Solution
First, identify the joint axes and imagine infinite lines along them.

![image|300](https://notes-media.kthiha.com/DH-Examples/fc72f290a13894c85e71af88efc64976.png)

Second, $\hat{z}_{i}$ coincides with joint axis $i$ and the $\hat{z}_{i-1}$-axis coincides with joint axis $i-1$.

![image|300](https://notes-media.kthiha.com/DH-Examples/aab0e0b64c9ee2efd5050aa4fb74e64c.png)

Third, place origin of link frame $i-1$ where mutually perpendicular line connecting $\hat{z}_{i-1}$ and $\hat{z}_{i}$ intersects $\hat{z}_{i-1}$

![image|300](https://notes-media.kthiha.com/DH-Examples/47ac8f047ca33626e26727714bed3e7e.png)

Fourth, $\hat{x}_{i-1}$ in direction of mutually perpendicular line pointing from joint axis $i-1$ to axis $i$.

![image|300](https://notes-media.kthiha.com/DH-Examples/a8eede8db7aa8c43434eb529525333ad.png)

Note the special cases that 
> **Adjacent Joint Axes Intersect**
> Choose $\hat{x}_{i-1}$ perpendicular to plane spanned by $\hat{z}_{i-1}$ and $\hat{z}_{i}$.

> **Adjacent Parallel Joint Axes**
> Choose mutually perpendicular line that is most physically intuitive and results in as many zero [[DH Parameter (Denavit-Hartenberg)|DH parameters]] as possible.

Fifth, $\hat{y}_{i}$ uniquely determined from cross-product $\hat{x} \times \hat{z}_{i}$

![image|300](https://notes-media.kthiha.com/DH-Examples/9b3a620d6399b53fb2b7708ae3e7cf08.png)

---
### Choice of Space and Body Frame
Note that choosing between space $\{ s \}$ and body $\{ b \}$ frames can yield different results.

![image|500](https://notes-media.kthiha.com/DH-Examples/de2d6a5f7429be74326fe61ea96ff300.png)

---