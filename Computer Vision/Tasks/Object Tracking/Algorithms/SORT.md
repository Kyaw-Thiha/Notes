# SORT 
#cv/object-tracking/algorithm/sort 

`Simple online and real-time tracking` is the pioneer of `tracking-by-detection`.

![SORT Algorithm|500](https://www.researchgate.net/publication/358134782/figure/fig2/AS:11431281350550980@1743777217913/Overview-of-the-object-tracking-SORT-algorithm.tif)

It has 4 components
- `Object Detection Model`
- `Motion Model`
- `Association Method`
- `Track Handling`

---
## Components of SORT

#### `1. Object Detection Model`

Although original `SORT` used `Faster R-CNN`, this can be replaced by any other object detection model (like `YOLOX`).

`SORT` performance is highly reliant on the hyperparameter $\text{detection score threshold}$ $(\det_{\tau})$.
A higher threshold reduces false positives, but increases false negatives.

---
#### `2. Motion Model`

`SORT` uses [[Kalman Filter]] to perform both `bounding box motion prediction` and `noise filtering`.

The state is modelled as
$$
x_{t} = [x,\ y,\ s,\ r,\ \dot{x},\ \dot{y},\ \dot{s}]
$$
where
- $x$ and $y$ represents bounding box `center coordinates`
- $s$ is the bounding box `scale` 
- $r$ is the bounding box `aspect ratio`
- $\dot{x}$ and $\dot{y}$ are `velocities of center coordinates`
- $\dot{s}$ is the `scale velocity`

Note that this model assumes that the bounding box aspect ratio does not change over time.

---
`Bayesian Inference`
1. `SORT` employs linear constant velocity model for track's motion prediction.
2. This motion prediction is used during `association step` to associate the track with detections.
3. `Bayesian Inference` is applied to combine information from both the `motion prediction` and `associated detection` to update the track's state.

`Bayesian Model`
In terms of [[Bayes Rule in ML|Bayesian Inference]],
$$
\text{Track's State}
\propto \text{Prior (motion prediction)}
\times \text{Likelihood (Detection)}
$$

It can be denoted as
$$
p(x_{t}|z_{t})
\propto p(z_{t} \mid x_{t})
\times p(x_{t} \mid p_{z_{1:t-1}})
$$

where
- $p(x_{t} \mid z_{1:t-1})$ is the `motion model prediction` $(\text{constant velocity})$
$$
p(x_{t} \mid z_{1:t-1}) 
= \int p(x_{t} \mid x_{t-1}) 
\ p(x_{t-1} \mid z_{1:t-1})
\ dx_{t-1}
$$
	where
	- $x_{t}$ is the `latent state` at time $t$
	- $z_{t}$ is the `measurement` at time $t$
- $p(z_{t} \mid x_{t})$ is the `measurement likelihood`
$$
p(z_{t} \mid x_{t}) 
= \mathcal{N} (z_{t} \mid Hx_{t},\ R)
$$
	where
	- Detection serves as the `measurment likelihood mean` $Hx_{t}$
	- Heuristics are used to approximate `measurement likehihood uncertainty` $R$

---
#### `3. Association Method`
`SORT` uses the [[Hungarian Algorithm]] to solve linear assignment problem in assigning detection to tracks.

- Association cost between track's prediction bounding box and a detection bounding box is equal to [[Intersection over Union (IoU)|Negative IoU]] $(\text{Intersection over Union})$.
- If $\text{IoU Similarity}$ is below threshold $\text{IoU}_{min}$, track cannot associated with corresponding detection as they are too far apart.
  This constraint is called `gates`.
- `Linear Assignment Solver` which can be represented as $\text{Association Cost Matrix}$
  where 
	- $\text{rows}$ correspond to $\text{tracks}$
	- $\text{columns}$ correspond to $\text{detections}$
- The solver returns 3 outputs:
	- A set of $\text{associated tracks}$ and $\text{detection}$ pairs
	- A set of $\text{unassociated tracks}$
	- A set of $\text{unassociated detections}$

---
#### `4. Track Handling Logic`
Track creation and deletion should follow objects that are entering, leaving, and returning to the scene over time.

- All `un-associated detections` are considered as potential new tracks with new identities
- To reduce false positive detections, each track needs to undergo a `probationary period` $(\text{e.g: 3 frames})$
  If at least one frame is missed during the `probationary period`, the potential new track is delete
- Existing tracks are terminated if they are not associated with any detection for $T_{lost}$ consecutive frames

---
## Algorithm
First we define the states as
- `Active`: Objects currently tracked with a successful association in the previous frame
- `Lost`: Objects currently tracked but failed to associate in one or more of the most recent frames $(\text{E.g: due to occlusion})$
- `Deleted`: Objects no longer tracked
- `New`: Objects in probationary period

Secondly, define following 
- $Det$ be an `object detection model`
- $KF$ be the [[Kalman Filter]]
- $IoU_{min}$ is the `association threshold`
- $\det_{\tau}$ is the `detection threshold`
- $T_{lost}$ is the `maximum track lost time`
- $HG$ is the [[Hungarian Algorithm]] returning associated pairs and un-associated singles
- $V$ is the `video sequence`
- $\hat{\tau}$ is the `predicted tracks`

Thirdly, we can define our pseudocode as
```python
Initialize tracks τ̂ ← ∅

for each frame (t, f_t) in video V do
    # 1. Object detection
    D ← Det(f_t, det_τ)

    # 2. Predict existing tracks
    T̂ ← ∅          # predicted track states

    for each track τ_k in alive(τ̂) do
        T̂.append( KF.predict(τ_k, t) )
    end for

    # 3. Data association
    C ← −IoU(D, T̂, IoU_min)

    (A, T_unassoc, D_unassoc) ← Hungarian(C)
    # A: matched (track_idx, detection_idx)
    # T_unassoc: unmatched tracks
    # D_unassoc: unmatched detections


    # 4. Update associated tracks
    for each (i_t, i_d) in A do
        τ ← τ̂[i_t]
        τ.update( KF.update(T̂[i_t], D[i_d]) )
        τ.lost ← 0
    end for


    # 5. Handle unassociated tracks
    for each i_t in T_unassoc do
        τ ← τ̂[i_t]
        τ.lost ← τ.lost + 1

        if τ.lost ≥ T_lost then
            τ̂.delete(τ)
        end if
    end for


    # 6. Create new tracks
    for each i_d in D_unassoc do
        τ̂.create( D[i_d] )
    end for
    
end for
```


---
## See Also
- [Main Reference Paper](https://arxiv.org/abs/2506.13457)
- [[DEEP SORT]]
- [[Object Tracking]]
- [[Kalman Filter]]