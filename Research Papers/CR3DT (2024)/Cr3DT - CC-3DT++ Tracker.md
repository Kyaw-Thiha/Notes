# CC-3DT++ Tracker

Enhanced tracking module in [[CR3DT (2024)|CR3DT]], extending CC-3DT with velocity-based association.

---

## Base Architecture

Built on `CC-3DT` (Cross-Camera 3D Tracking) with three key improvements.

**Components**:
- **Appearance embeddings** 
  [[Quasi-Dense Similarity Learning]] with contrastive loss
- **State estimation** 
  [[Kalman Filter]] for motion prediction
- **Data association** 
  Greedy matching based on affinity matrix

---

## Data Association

Association between detections $D_t$ and active tracks $T_t$ uses an affinity matrix:

$$
A(T_t, D_t) = w_{deep} \cdot A_{deep}(T_t, D_t) + w_{motion} \cdot A_{motion}(T_t, D_t) \cdot A_{loc}(T_t, D_t)
$$

where:
- $A_{deep}$: Appearance embedding similarity
- $A_{motion}$: Motion correlation (modified formulation)
- $A_{loc}$: Location correlation (spatial proximity)
- $w_{deep} + w_{motion} = 1$

---

## Key Innovation: Velocity Similarity

### Novel Motion Correlation

$$
a_{motion}(\tau_t, d_t) = a_{vel} \cdot a_{centroid} + (1 - a_{vel}) \cdot a_{pseudo}
$$

where:
- $a_{centroid}$: Centroid position correlation
- $a_{pseudo}$: Pseudo-velocity from position differences (original CC-3DT)
- $a_{vel}$: **NEW velocity similarity weight**

### Velocity Similarity Weight

$$
a_{vel}(\tau_t, d_t) = \exp\left(-\frac{1}{r}|v_{\tau_t} - v_{d_t}|\right)
$$

where:
- $v_{\tau_t} = [v_x, v_y, v_z]^T$: [[Kalman Filter]] predicted velocity
- $v_{d_t} = [v_x, v_y, v_z]^T$: Detector output velocity
- $r$: Tunable radius parameter

**vs Original CC-3DT**:
- **Original**: Cosine similarity on pseudo-velocities (position deltas)
- **CC-3DT++**: Exponential similarity on actual velocities from RADAR
- **Impact**: 6.8% reduction in ID switches

---
## Why It Works

**Leverages RADAR's strength**:
- RADAR provides direct, accurate velocity measurements
- Original CC-3DT used pseudo-velocities (derived from positions)
- Direct velocity comparison more reliable than direction-based cosine similarity

**Tuning matters**:
- Default CC-3DT threshold too conservative for this task
- Motion correlation should dominate with reliable velocity data

---

## See Also
- [[CR3DT (2024)]] 
- [[Kalman Filter]] 
- [[Quasi-Dense Similarity Learning]]