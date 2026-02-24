`Limitation of Mahalanobis Distance`

The key property of `Mahalanobis Distance` is
$$
d^2_{M} = (z - \hat{z})^T \ S^{-1} \ (z - \hat{z})
$$
where
- $z$ is the `actual measurement` of track from detector 
- $\hat{z}$ is the `predicted measurement` of track from [[Kalman Filter]]
- $(z -  \hat{z})$ is the `innovation`
- $S$ is the `innovation covariance`


Suppose a track becomes `lost`.
- Then, `Kalman Filter` keeps predicting it forward
- This causes the `covariance` $S$ to grow larger over time.
- Larger $S$ $\implies$ Smaller $S^{-1}$
- The `Mahalanobis Distance` shrinks even for bad matches.
- This makes lost tracks cheaper to associate with compared to active tracks.

---
`Numeric Example`

An active track with error $z - \hat{z} = 5$ and uncertainty $S=1$ has
$$
d^2_{M} = \frac{25}{1} = 25
$$
while a lost track with error $z - \hat{z} = 20$ and uncertainty $S=100$ has 
$$
d^2_{M} = \frac{400}{100} = 4
$$
Hence, despite being $4 \text{\ times}$ further away, the lost track is cheaper to match.

---
`Cascaded Association`

To address the limitation of `Mahalanobis Distance`, `Deep SORT` uses cascaded assocation based on track's time lost.

This means Deep SORT does not associate all tracks at once.
Instead, it
- Groups tracks by how long they've been lost
- Matches fresh tracks first
- Only later tries to match older (more uncertain) tracks

---
## See Also 
- [[DEEP SORT]]
