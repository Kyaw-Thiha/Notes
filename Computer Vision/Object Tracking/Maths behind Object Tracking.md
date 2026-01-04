# Maths behind Object Tracking
#cv/object-tracking/maths 

This is the mathematical model for [[Object Tracking]].

![Object Tracking|400](https://miro.medium.com/v2/resize:fit:1000/1*SOGkeT9Yi1Zscl0b2ydwRQ.gif)

`Objects`
Let $K$ be a set of identities of objects that are present in the video over a discrete time interval $\{ 1, \ 2, \ \dots, \ T \}$.

At any time $t$,
$$
X_{t} = \{ x_{t}^k \mid k \in K_{t} \}
$$
where
- $X_{t}$ is a set of `visible objects`
- $x_{t}^k$ is the `ground truth` of object with identity $k$ at time $t$
- $K_{t} \subseteq K$ is a set of `visible objects at time` $t$

`Trajectories`
The objective is to compute the trajectories of all objects in the video
$$
\tau^k = \{ x_{t}^k \mid t \in T_{k} \}
$$
where 
- $t \in T_{k}$ is the timestamp
- $\tau^k$ is the trajectory of object $k$

The set of all such trajectories is
$$
\tau = \{ \tau^k \mid k \in K \}
$$

Note that the trajectory $\tau^k$ can be fragmented due to being occluded or out of view.

`Multi-Object Tracker`
A `MOT Tracker` predicts video tracks $\hat{\tau}$, aiming to minimize error between predicted $\hat{\tau}$ and ground truth $\tau$.

We can denote
- $\hat{\tau}^k$ as `predicted track`
- $\hat{x}^k_{t}$ as `predicted detection`

---
## See Also
- [[Object Tracking]]
