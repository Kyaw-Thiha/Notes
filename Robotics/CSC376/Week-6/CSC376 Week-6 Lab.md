# Week-6 Lab
## Question-1
### Question-$(a)$
> Find inverse-kinematics when the end-effector position is $p=(p_{x}, \ p_{y}, \ 2L)^{T}$

![image|200](https://notes-media.kthiha.com/CSC376-Week-6-Tutorial/1ef34d893258ffc292494f00c561b87a.png)

### Solution
Drawing a right triangle on the `x-y plane`, we get $\tan(\theta_{1}) = \frac{p_{y}}{p_{x}}$.
$$
\begin{align}
\tan(\theta_{1}) &= \frac{p_{y}}{p_{x}} \\[6pt]
\arctan(\tan(\theta_{1})) &= \arctan(\frac{p_{y}}{p_{x}}) \\[6pt]
\theta_{1} &= \arctan(\frac{p_{y}}{p_{x}}) \\[6pt]
\end{align}
$$
To bound $\theta_{1} \in \left( -\frac{\pi}{2}, \ \frac{\pi}{2} \right)$, we use $\theta_{1} = \arctan2\left( \frac{p_{y}}{p_{x}} \right)$.

---
### Question-$(b)$
> The end-effector position is $p=(p_{x}, \ p_{y}, \ p_{z})^{T}$.

![image|200](https://notes-media.kthiha.com/CSC376-Week-6-Tutorial/3e088f7a0a0b5e578130b1b20a3ba7e5.png)

We have $\theta_{1} = \arctan2\left( \frac{p_{y}}{p_{x}} \right)$ by assuming a fixed $\theta_{2}$.
We can then get $p_{y} = p_{x} \arctan^{-1}2(\theta_{1})$. Note that we could use the x-component instead.

We can then get that $\theta_{2} = p_{x} \arctan^{-1}2(\theta_{1})$.

jes pythagorus

---
### Question-$(c)$
> The end-effector position is $p=(p_{x}, \ p_{y}, \ p_{z})^{T}$ and orientation $R = e^{[z]\alpha}$ where $z=(0,0,1)^{T}$ and $\alpha=[0, 2\pi]$.

![image|200](https://notes-media.kthiha.com/CSC376-Week-6-Tutorial/9e227f51c900d8c90d0b161690710a33.png)

We can use $\theta_{1}$ to get us the $(p_{x}, \ p_{y})$, $\theta_{2}$ to get the $\alpha$ and $\theta_{3}$ to get the $p_{z}$.
Note that looking from bird-eye-view, we can see that $\alpha = \theta_{1} + \theta_{2}$ since $\alpha$ is the angle between space and body frame.

> This yields as $\theta_{1} = \arctan 2\left( \frac{p_{y}}{p_{x}} \right),$ $\theta_{2} = \alpha - \theta_{1}$ and $\theta_{3} = p_{z}$.

---
## Question-2
For a planar 3R robot of link lengths $l_{1}, l_{2},$ and $l_{3}$, the position $(p_{x}, p_{y})^{T}$ and orientation $\alpha$ of the end-effector is given. Note that the orientation $\alpha$ is defined as the angle between the unit vectors $\hat{x}_{s}$ and $\hat{x}_{b}$. 

Find its inverse kinematics solution.

![image|200](https://notes-media.kthiha.com/CSC376-Week-6-Tutorial/223b112196d9d28a6028b4c2cd9a9818.png)

### Solution
First note that 
$$
\alpha = \theta_{3} + \theta_{2} + \theta_{1}
$$ 

This implies that we can reduce our problem to two parameters $\theta_{3}$ and $\theta_{2}$, and use two equations.

We need to first start with $\theta_{3}$ since its the nearest to the end-effector.
Looking at axis along $l_{2}$, we have $\tan(\theta_{3}) = \frac{y_{b}}{y_{x}}$ which yields us $\theta_{3} = \arctan2\left( \frac{y_{b}}{x_{b}} \right)$.

We can use this to get 
- $y_{3} = x_{b} \ \arctan^{-1}2(\theta_{3})$ 
- $x_{3} = \frac{y_{b}}{\arctan^{-1}2(\theta_{3})}$ 

Likewise, we can continue that kinematics to $\theta_{2}$ whereby 
$$
\theta_{2} = \arctan2\left( \frac{y_{3}}{x_{3}} \right)
= \arctan2\left( \frac{x_{b} \arctan^{-1} 2(\theta_{3})}{y_{b}/\arctan ^{-1} 2(\theta_{3})} \right)
$$

---
## Question-3
Consider the following RRP serial manipulator. For a set of given end-effector coordinates $p=(x,y,z)^{T}$ in the fixed frame $\{ 0 \}$, find the joint variables $(\theta_{1}, \ \theta_{2}, \ \theta_{3})^{T}$ in the form of an explicit formula. 


![image|200](https://notes-media.kthiha.com/CSC376-Week-6-Tutorial/5f893b032b58e0edf59048bc7990ca0e.png)

Assume that all the $\theta$ values are defined to be $0$ along their local $x$ axis.

### Solution
First
$$
\sin(45deg) = -\frac{z}{\theta_{3}}
$$
With $\sin(45deg) = \frac{1}{\sqrt{ 2 }}$, we get
$$
\theta_{3} = -\sqrt{ z }
$$

This decomposes the robot into a 2R robot when viewed through the bird-eye-view.

---