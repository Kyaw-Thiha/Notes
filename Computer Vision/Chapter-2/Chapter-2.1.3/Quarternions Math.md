#math #cv/transformations/3d/rotation/quarternion

**Quarternion**: $a + b.\hat{i} + c.\hat{j} + d.\hat{k}$ 
where
- $a$ is the real part
- $b.\hat{i} + c.\hat{j} + d.\hat{k}$ is the imaginary / vector part

## Main Properties 
Let $\vec{p}$, $\vec{q}$ and $\vec{s}$ be quarternions.
### Scalar Multiplication
$$
k.\vec{p} = k.(a, b, c, d)
= (ka, kb, kc, kd)
$$

### Vector Multiplication
#### Scalar Part
Multiplying the scalar part of quarternion, to another arbitrary quarternion act the same as scalar multiplication.
$$
\begin{aligned}
(k, 0, 0, 0).\vec{p}
&= (k, 0, 0, 0).(a, b, c, d) \\
&= (ka, kb, kc, kd)
&= k.\vec{p}
\end{aligned}
$$

#### Vector Part
#### Multiplication Table of Quarternions
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
- $i, j, k$ are imaginary number $\implies$ their square is -1
- Their product follows a circular pattern
$$ i \to j \to k \to i \implies ij=k, jk=i, ki =j $$
- Change order of product $\implies$ Change sign
$$
ij = -ji, jk = -kj, ki = -ik
$$
#### Long Formula
$$
\begin{aligned}
& (a+bi+cj+dk)⋅(w+xi+yj+zk) \\[1ex]
&=(aw−bx−cy−dz) \\
&+(ax+bw+cz−dy).\vec{i} \\
&+(ay−bz+cw+dx).\vec{j} \\
&+(az+by−cx+dw).\vec{k} 
\end{aligned}
$$
### Non-Commutativity
Based on the vector multiplication (the vector part), we can conclude that quarternions are not commutative.
Proof is too verbose to write + I am lazy
$$
\vec{p}.\vec{q} \neq \vec{q}.\vec{p}
$$
### Associativity
$$
\vec{p}.(\vec{q}.\vec{r}) = (\vec{p}.\vec{q}).\vec{r}
$$

---
## Scalar-Vector Representation
Quarternions can be represented as scalar and vector parts.
$$
\begin{aligned}
\vec{p} &= a + x.\hat{i} + y.\hat{j} + z.\hat{k} \\[1ex]
&= a + \vec{u} \\[1ex]
&= (a, \vec{u})
\end{aligned}
$$

## Vector Multiplication (Short Formula)
$$
(a+\vec{u})⋅(b+\vec{v})
=(ab−⟨\vec{u},\vec{v}⟩) + (av+bu+\vec{u}×\vec{v})
$$
where 
- $⟨\vec{u},\vec{v}⟩$ is the 3D dot product 
-  $\vec{u} × \vec{v}$ is the 3D cross product

---
## Deriving Conjugate & Length
When computing the square of a quarternion,
$$
\begin{aligned}
\vec{q} &= (a + \vec{u}).(a + \vec{u}) \\
&= (a^2−⟨\vec{u},\vec{u}⟩) + (a.\vec{u}+a.\vec{u}+\vec{u}×\vec{u}) \\
&=(a^2−⟨\vec{u},\vec{u}⟩) + 2a.\vec{u}
\end{aligned}
$$
When we change the sign of the vector component,
$$
\begin{aligned}
&(a + \vec{u}).(a - \vec{u}) \\[1ex]
&= a^2−⟨\vec{u},−\vec{u}⟩)+(a.\vec{u}−a.\vec{u} + \vec{u}×(−\vec{u})) \\[1ex]
&=(a^2+⟨\vec{u},\vec{u}⟩) + 0
\end{aligned}
$$
By $⟨\vec{u},\vec{u}⟩=|\vec{u}|^2$, 
$$
\begin{aligned}
(a + \vec{u}).(a - \vec{u}) 
&= a^2 + |\vec{u}|^2\\[1ex]
&= a^2 + x^2 + y^2 + z^2
\end{aligned}
$$
which is the Eucliean distance for 4D vector $(a, x, y, z)$ squared.

### Length of Quarternion
$$
\begin{aligned}
\text{Length of } \vec{q} &= |\vec{q}| \\[2ex]
&= \sqrt{a^2 + ⟨\vec{u},\vec{u}⟩} \\[2ex]
&= \sqrt{ a^2 + x^2 + y^2 + z^2 }
\end{aligned}
$$

### Conjugate
Negate the vector component
Let $\vec{q} = a + \vec{u}$ 
Then, $\vec{q^c} = a - \vec{u}$

### Length & Conjugate
$$
|\vec{q}|^2 = \vec{q}.\vec{q^c}
$$
   
### Commutativity of Conjugate Multiplication

$$
\begin{aligned}
|q^c|^2 
&= a^2+⟨−u,−u⟩ \\
&=a^2+(−x)^2+(−y)^2+(−z)^2 \\
&=a^2+x^2+y^2+z^2 \\
&=a^2+⟨u,u⟩ \\
&=|q|^2
\end{aligned}
$$
We can use the above formula and length formula to prove that 
$$
\begin{aligned}
\vec{q}.\vec{q^c} 
&= |q|^2 \\[1ex]
&= |(q^c)^c|^2 \\[1ex]
&= |q^c|^2  \\[1ex]
&= \vec{q^c}.\vec{q}
\end{aligned}
$$

## Length between 2 Quarternions
First we need to derive that $(\vec{p}.\vec{q})^c = \vec{q^c}.\vec{p^c}$
Note that it also generalises as $(p_{1}⋅⋯⋅p_{n})^c = p_{n}^c⋅⋯⋅p_{1}^c$ ^conjugate-over-multiplication-of-quarternions

Using the above property,
$$
\begin{aligned}
|p⋅q|^2 
&= (p⋅q)⋅(p⋅q)^c \\[1ex]
&=(p⋅q)⋅(q^c⋅p^c) \\[1ex]
&=p⋅(q⋅q^c)⋅p^c \\[1ex]
&=p⋅|q|^2⋅p^c \\[1ex]
&=|q|^2.p⋅p^c \\[1ex]
&=|q|^2.|p|^2
\end{aligned}
$$

It also generalises as $|p_{1}⋅⋯⋅p_{n}| =|p_{1}|⋅⋯⋅|p_{n}|$

---
## Scalar Component via Conjugation
$$
\vec{q} + \vec{q^c} 
= (a + \vec{u}) + (a - \vec{u})
= 2a
$$
Hence, we can separate the scalar component by $\frac{\vec{q} + \vec{q^c}}{2}$

## Vector Component via Conjugate
$$
\vec{q} - \vec{q^c} 
= (a + \vec{u}) - (a - \vec{u})
= 2.\vec{u}
$$
Hence, we can separate the vector component by $\frac{\vec{q}-\vec{q^c}}{2}$

## Dot Product via Conjugation
Lets investigate into product between a vector, and a conjugate of another vector.
Let $\vec{p} = a + \vec{u} = (a, b, c, d)$ and $\vec{q} = w + \vec{v} = (w, x, y, z)$

$$
\begin{aligned}
&\vec{p}.\vec{q^c} \\[1ex]
&= (a + \vec{u}).(w - \vec{v}) \\[1ex]
&= (aw + \langle \vec{u}, \vec{v} \rangle)
+ (-a.\vec{v} + w.\vec{u} - \vec{u} \times \vec{v})
\end{aligned}
$$

&nbsp;

Note that the scalar component is $a.w + \langle \vec{u}, \vec{v} \rangle$.
$$
\begin{aligned}
&a.w + \langle \vec{u}, \vec{v} \rangle \\[1ex]
&= a.w + b.x + c.y + d.z \\[1ex]
&= \langle \vec{p}, \vec{q} \rangle
\end{aligned}
$$
which is also the cross product of $\vec{p}$ and $\vec{q}$

&nbsp;

Thus, we can use the [[#Scalar Component via Conjugation]] formula to derive that
$$
\langle \vec{p}, \vec{q} \rangle 
= \frac{\vec{p}. \vec{q^c} + (\vec{p} + \vec{q^c})^c}{2}
$$
Then, we can use [[#^conjugate-over-multiplication-of-quarternions | Conjugate over multiplication of quarternions]] formula to conclude that
$$
\langle \vec{p}, \vec{q} \rangle 
= \frac{1}{2} .(\vec{p}.\vec{q^c} + \vec{q}.\vec{p^c})
$$
### Special Case of Vector Quarternion
Suppose the 2 quarternions only consist of its vector components such that $\vec{p} = 0 + \vec{u}$ and $\vec{q} = 0 + \vec{v}$
&nbsp;
Then, scalar component of $\vec{p}.\vec{q^c}$ is $bx + cy + dz$, which is equal to $\langle \vec{v}, \vec{u} \rangle$.
Hence,
$$
\begin{aligned}
\langle \vec{p}, \vec{q} \rangle
&= bx + cy + dz \\[1ex]
&= \langle \vec{u}, \vec{v} \rangle \\[1ex]
&= \frac{1}{2}.(\vec{v}.(-\vec{u}) + \vec{u}.(-\vec{v})) \\[1ex]
&= -\frac{1}{2}.(\vec{v}.\vec{u} + \vec{u}.\vec{v})
\end{aligned}
$$

## Cross Product via Conjugation
Similarly, consider 2 quarternions with only vector component such that $\vec{p} = \vec{v}$ and $\vec{q} = \vec{u}$.
Then, $\vec{v}.\vec{u} = (-\langle \vec{v}, \vec{u} \rangle) + (\vec{v} + \vec{u})$ by [[#Vector Multiplication (Short Formula)]]

&nbsp;

The vector component is the $\vec{v} + \vec{u}$.
Thus, we can use [[#Vector Component via Conjugate]] to derive that
$$
\begin{aligned}
\vec{v} \times \vec{u} 
&= \frac{1}{2}.(\vec{v}.\vec{u} - (\vec{v}.\vec{u})^c) \\[2ex]
&= \frac{1}{2}.(\vec{v}.\vec{u} - \vec{u^c}.\vec{v^c}) \\[2ex]
&= \frac{1}{2}.(\vec{v}.\vec{u} - (-\vec{u}).(-\vec{v})) \\[2ex]
&= \frac{1}{2}.(\vec{v}.\vec{u} - \vec{u}.\vec{v})
\end{aligned}
$$

## Proving Quarternion's Relation to Rotation
Consider the following formula 
$$
\vec{q}.\vec{v}.\vec{q^c}
$$
where $\vec{q}$ is a unit quarternion and $\vec{v}$ is a 3D vector

### Prove that $\vec{q}.\vec{v}.\vec{q^c}$ is vector quarternion
Let $\vec{u} = (\vec{q}.\vec{v}.\vec{q^c})$.
Then, using[[#^conjugate-over-multiplication-of-quarternions | (1): Conjugate over multiplication of quarternions]] 
$$
\begin{aligned}
\vec{u^c} 
&= (\vec{q}.\vec{v}.\vec{q^c})^c  \\[2ex]
&= (\vec{q^c})^c.\vec{v^c}.\vec{q^c} && \text{by (1)} \\[1ex]
&= \vec{q}.(-\vec{v}).\vec{q^c} 
&& \text{by conjugate of 3D vector} \\[2ex]
&= -\vec{u} && ,\text{as wanted}
\end{aligned}
$$

### Length Preservation
Consider the length of $\vec{u}$
$$
\begin{aligned}
|\vec{u}| &= |\vec{q}|.|\vec{v}|.|\vec{q^c}| \\[2ex]
&= |\vec{q}|.|\vec{v}|.|\vec{q}| \\[2ex]
&= |\vec{q}|^2.|\vec{v}| \\[2ex]
\end{aligned}
$$
Since $\vec{q}$ is a unit quarternion, its length is 1.
Hence, $|\vec{u}| = |\vec{v}|$ implying that the transformation $\vec{q}.\vec{v}.\vec{q^c}$ preserves length.

### Dot Product Preservation
Let $\vec{u}$ and $\vec{v}$ be arbitrary 3D vectors.
Then, using the [[#Cross Product via Conjugation]],
$$
\begin{aligned}
\langle 
	\vec{p}.\vec{v}.\vec{q^c}, \vec{q}.\vec{u}.\vec{q^c} 
\rangle
&= -\frac{1}{2}.((p.v.p^c)⋅(p.u.p^c)+(p.u.p^c)⋅(p.v.p^c)) \\[2ex]
&= -\frac{1}{2}.(p.v.(p^c.p).u.p^c + p.u.(p^c.p)v.p^c) \\[2ex]
&= -\frac{1}{2}(p.v⋅|q|^2⋅u.p^c+p.u⋅|q|^2⋅v.p^c) \\[2ex]
&= -\frac{1}{2}(p.v.u.p^c+p.u.v.p^c) \\[2ex]
&= -\frac{1}{2}(p.(v.u).p^c+p.(u.v).p^c) \\[2ex]
&= -p.\frac{1}{2}(p(v.u)p^c + p(u.v)p^c) \\[2ex]
&= \vec{p}.\langle \vec{v}, \vec{u} \rangle.\vec{p^c} \\[2ex]
&= \langle \vec{v}, \vec{u} \rangle.\vec{p}.\vec{p^c} \\[2ex]
&= \langle \vec{v}, \vec{u} \rangle.|\vec{p}|^2 \\[2ex]
&= \langle \vec{v}, \vec{u} \rangle
\end{aligned}
$$
This implies that the transformation $\vec{q}.\vec{v}.\vec{q^c}$ preserves angle.
Before we can conclude that this is a rotation, we need to check if it preserves orientation, as it could still be reflection.

### Cross Product Preservation
We can check if orientation is preserved or not by applying cross product of transformation of first 2 vector components $\hat{i}$ and $\hat{j}$, and see if the result is transformation of $\hat{k}$ or $-\hat{k}$.
Using [[#Multiplication Table of Quarternions | (1) Multiplication Table of Quarternions]] 
$$
\begin{aligned}
(\vec{p}.\hat{i}.\vec{p^c}) \times (\vec{p}.\hat{j}.\vec{p^c})
&= \frac{1}{2}.
(p.\hat{i}.p^c.p.\hat{j}.p^c−p.\hat{j}.p^c.p.\hat{i}.p^c) 
\\[2ex]
&= \frac{1}{2}.(\vec{p}.\hat{i}.\hat{j}.\vec{p^c} -
\vec{p}.\hat{i}.\hat{j}.\vec{p^c}) \\[2ex]
&= \frac{1}{2}.(\vec{p}.\hat{k}.\vec{p^c} -
\vec{p}.(-\hat{k}).\vec{p^c})
&& \text{by (1)} \\[2ex]
&= \frac{1}{2}.(\vec{p}.\hat{k}.\vec{p^c} +
\vec{p}.\hat{k}.\vec{p^c}) \\[2ex]
&= \vec{p}.\hat{k}.\vec{p^c}
&& \text{as wanted}
\end{aligned}
$$
This implies that orientation is preserved.

Using 
- [[#Prove that $ vec{q}. vec{v}. vec{q c}$ is vector quarternion |Transformation is vector quarternion]]
- [[#Length Preservation]]
- [[#Dot Product Preservation]]
- [[#Cross Product Preservation]]
we can conclude that $\vec{v} \to p.\vec{v}.p^c$ is a rotation.


## Read Also
- [Blog Post this Page is written from](https://lisyarus.github.io/blog/posts/introduction-to-quaternions.html)
