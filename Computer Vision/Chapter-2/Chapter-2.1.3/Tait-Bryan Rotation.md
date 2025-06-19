# Tait-Bryan Rotation
#math #cv #aviation  

A special [[Davenport Rotation]] in which first and third axes are perpendicular to each other.
 
**Tait-Bryan Rotations**: 
(x-y-z, y-z-x, z-x-y, x-z-y, z-y-x, y-x-z)

A practical example is the yaw-pitch-roll (YPR) rotation used in #aviation. 

## Yaw-Pitch-Roll Example

$$
Roll(\phi) = R_{x}(\phi) = 
\begin{bmatrix}
1 & 0 & 0 \\
0 & \cos(\phi) & -\sin(\phi) \\
0 & \sin(\phi) & \cos(\phi)
\end{bmatrix}
$$

$$
Pitch(\phi) = R_{y}(\theta) = 
\begin{bmatrix}
\cos(\theta) & 0 & \sin(\theta) \\
0 & 1 & 0 \\
-\sin(\theta) & 0 & \cos(\theta)
\end{bmatrix}
$$

$$
Yaw(\psi) = R_{z}(\psi) = 
\begin{bmatrix}
\cos(\psi) & -\sin(\psi) & 0 \\
\sin(\psi) & \cos(\psi) & 0 \\
0 & 0 & 1
\end{bmatrix}
$$
$$
\begin{aligned}
R &= Yaw(\psi).Pitch(\theta).Roll(\phi) \\
  &= R_{z}(\psi).R_{y}(\theta).R_{x}(\phi)
\end{aligned}
$$

## Graphical Representation of YPR
![Plane in Yaw-Pitch-Roll](./attachments/Plane_with_embedded_axes.png)

![Plane in Yaw-Pitch-Roll](./attachments/Yaw_Axis_Corrected.png)

## Graphical Representation of Yaw
![Yaw](./attachments/Aileron_yaw.gif)

## Graphical Representation of Pitch
![Yaw](./attachments/Aileron_pitch.gif)

## Graphical Representation of Roll
![Yaw](./attachments/Aileron_roll.gif)
## See Also
- [[Davenport Rotation]]
- [[Euler Rotation]]

## Read Also
- [Wikipedia](https://en.wikipedia.org/wiki/Davenport_chained_rotations#Tait%E2%80%93Bryan_chained_rotations)
- [Aircraft Principal Axis Wikipedia](https://en.wikipedia.org/wiki/Aircraft_principal_axes)
