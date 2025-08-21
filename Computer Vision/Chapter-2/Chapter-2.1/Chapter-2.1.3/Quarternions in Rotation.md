#math #cv/transformations/3d/rotation/quarternion
# Quarternions in Rotation

Quarternions are 4D vectors $(a, b, c, d)$, that comprises of the scalar component $a$ and vector/imaginary component $\vec{u} = (a, b, c)$.
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
R(\hat{n}, \theta) = 
\text{I} + 
\sin(\theta).[\hat{n}]_{x} +
(1-\cos(\theta))[\hat{n}]^2_{x}
$$
and a quarternion can be represented as
$$
\vec{q} = (\vec{v}, w) 
= \left( \sin\left( \frac{\theta}{2} \right).\hat{n}, 
\cos\left( \frac{\theta}{2} \right) \right)
$$
Substituting it into Rodrigues Formula,
$$
\begin{aligned}
R(\hat{n}, \theta) 
&= \text{I} + 
\sin(\theta).[\hat{n}]_{x} +
(1-\cos(\theta))[\hat{n}]^2_{x} \\[2ex]
&= I + 2w[\vec{v}]_{\text{x}} + 2.[\vec{v}]^2_{\text{x}}
\end{aligned}
$$

## Benefit over Axis-Angle
The above equation, highlight the main benefit of quarternion over axis-angle representation:  

$$
\text{no trig func in evaluation} \implies
\text{computationally efficient}
$$

Rotating using axis-angle requires ~30 computations, with repetitive trigonometric functions.

On the other hand, quarternions only requires ~15-20 computations, with trigonometric functions only required for initial conversion.

Thus, quarternions are better for continuous 3D rotation values.

## Slerp
Slerp (Spherical Linear Interpolation) is a method to **smoothly** interpolate between 2 orientation or rotations.

```python
import numpy as np

def normalize(q):
    return q / np.linalg.norm(q)

def slerp_manual(q0, q1, t):
    q0 = normalize(q0)
    q1 = normalize(q1)

    dot = np.dot(q0, q1)

    # If the dot product is negative, 
    # negate one quaternion to take the shorter path
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
		# to ensure numerical stability (when sin(θ) ≈ 0)
		# perform LERP (linear interpolation) as fallback
        result = q0 + t * (q1 - q0)
        return normalize(result)

	# If sin(θ) is below threshold
    theta_0 = np.arccos(dot)
    theta = theta_0 * t

    q2 = normalize(q1 - q0 * dot)
    return q0 * np.cos(theta) + q2 * np.sin(theta)

# Example
q0 = np.array([0, 0, 0, 1])
q1 = np.array([0.7071, 0, 0, 0.7071])
t = 0.5

result = slerp_manual(q0, q1, t)
print(result)
```

## Math behind Quarternions
If you want to learn more about all the properties, and their proofs, as well as the proof of [[Quarternions Math#Proving Quarternion's Relation to Rotation | why quarternion can be used to represent 3D rotations]] , you can read [[Quarternions Math |here]]

## Read More
- [Blog Post this Page is written from](https://lisyarus.github.io/blog/posts/introduction-to-quaternions.html)
- [Short Video by 3Blue1Brown](https://youtu.be/zjMuIxRvygQ?si=d1w75o6XHkczPmGj)
- [Long Video by 3Blue1Brown (Visualization)](https://youtu.be/d4EgbgTm0Bg?si=RJHW44zfiVgh5JQH)

