# Joint 
[[Joint|Joints]] connect [[Rigid Body|rigid bodies]] called links to each other.

![200](https://www.mdpi.com/energies/energies-14-06690/article_deploy/html/images/energies-14-06690-g001.png)

---
## Joint Types

![400](https://stevengong.co/attachments/Pasted-image-20211112090327.png)

![image|400](https://notes-media.kthiha.com/Joint/cf66efb6efe5557bb973c6969f8a6124.png)

---
### Revolute(R) Joint

![300](https://community.sw.siemens.com/sfc/servlet.shepherd/version/renditionDownload?rendition=THUMB720BY480&versionId=068Vb00000QZJ2S&operationContext=CHATTER&contentId=05TVb00000VOCL0&page=0)

- constraints: $5$
- freedom: $1$

It places $5$ constraints on second [[Rigid Body|rigid body]] relative to the first.
Therefore, the second body has only $1$ [[Degrees of Freedom(DOF)|degrees of freedom]], given by the angle of the [[Joint|revolute joint]].

---
### Prismatic(P) Joint

![300](https://sp-ao.shortpixel.ai/client/to_auto,q_glossy,ret_img,w_392,h_170/https://learnchannel-tv.com/wp-content/uploads/2020/10/Kinematics-Linked-joints.gif)

Also known as a [[Joint|linear joint]], it also has one [[Degrees of Freedom(DOF)|degrees of freedom]].

---
### Universal Joint

![300](https://d2t1xqejof9utc.cloudfront.net/screenshots/pics/5861891c555e080697c2864fb810c580/medium.gif)

- constraints: $4$
- freedom: $2$

---
### Spherical Joint

![200](https://i.pinimg.com/originals/e1/49/26/e149264f432ad850af306ff82ae3f6f4.gif)

- constraints: $3$
- freedom: $2$

The [[Joint|spherical joint]] has $2$ degrees of freedom of a universal joint, plus spinning about the axis.

---
## Deriving Grubler's Formula

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

---
## See Also
- [Modern Robotics Video](https://modernrobotics.northwestern.edu/nu-gm-book-resource/2-2-degrees-of-freedom-of-a-robot/)
- [[Joint]]
- [[Grubler's Formula]]
- [[Stewart Platform]]
- [[Degrees of Freedom(DOF)]]
- [[Rigid Body]]
