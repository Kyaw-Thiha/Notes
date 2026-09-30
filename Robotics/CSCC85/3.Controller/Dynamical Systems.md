# Dynamical Systems
[[Dynamical Systems|Dynamical systems]] are systems that change through time. 
We are going to be representing this with a vector of variables.

$$
\hat{x}_{t} = \begin{bmatrix}
\text{position} \\[6pt]
\text{velocity} \\[6pt]
\text{gas} \\[6pt]
\text{batter level} \\[6pt]
\vdots
\end{bmatrix}
$$

---
## Modelling
To solve a problem, we needs to model things. But this is not a real thing, so there could be differences.
![image|300](https://notes-media.kthiha.com/Dynamical-Systems/e7b761d1392cba152f31983bcc91e7e0.png)

---
### Simple Model
$$
\frac{d\bar{s}}{dt} = \begin{bmatrix}
0 & 1 \\[6pt]
0 & 0
\end{bmatrix}
\begin{bmatrix}
x \\[6pt] v_{x}
\end{bmatrix}
$$
In here, we have $\frac{dx}{dt} = v_{x}$ and $\frac{dv_{x}}{dt}=\emptyset$.

---
### Better Model
$$
\frac{d\bar{s}}{dt}
= \begin{bmatrix}
0 & 1 & 0 \\[6pt]
0 & 0 & 1 \\[6pt]
0 & 0 & 0 
\end{bmatrix}
\begin{bmatrix}
x \\[6pt]
v_{x} \\[6pt]
a_{x} \\[6pt]
\end{bmatrix}
$$
The [[Controller|controller]]'s job is to use the state variables in matrix $A$ to control the state it wants to maintain.

---
