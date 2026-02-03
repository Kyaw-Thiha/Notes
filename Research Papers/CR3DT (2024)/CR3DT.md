# CR3DT (2024)

`CR3DT (Camera-RADAR 3D Detection and Tracking)` is a sensor fusion framework that combines camera and RADAR data in `Bird's-Eye-View (BEV) space` for 3D object detection and multi-object tracking in autonomous driving.

![CR3DT|500](https://notes-media.kthiha.com/CR3DT/2dc06fea2f58bf6120ee44d511dfd041.png)

---

## Architecture Overview

`CR3DT` extends [[BEVDet (2022)|BEVDet]] with two main contributions:
1. [[CR3DT - Camera-RADAR Fusion Strategy|Camera-RADAR Fusion]]: Two-stage fusion strategy in BEV space
2. [[CR3DT - CC-3DT++ Tracker|CC-3DT++ Tracker]]: Enhanced tracking with velocity-based association

### Pipeline
1. `Camera Stream`
   $6$ views $\to$ [[ResNet|ResNet-50]] → [[Lift, Splat, Shoot]] $\to$ BEV features $(64 \times 128 \times 128)$
2. `RADAR Stream` 
   5 sensors $to$ [[PointPillars]]-style encoding $\to$ BEV features $8 \times 128 \times 128$
3. `Fusion`: [[CR3DT - Camera-RADAR Fusion Strategy|Two-stage concatenation]] in BEV space
4. `Detection`: [[CenterPoint]] head → 3D boxes + velocities
5. `Tracking`: [[CR3DT - CC-3DT++ Tracker|Velocity-based association]] → Track IDs

---
## See More
- [Paper](https://arxiv.org/abs/2403.15313)
- [Code](https://github.com/ETH-PBL/CR3DT)
- Architecture Details: 
  - [[CR3DT - Camera-RADAR Fusion Strategy]]  
  - [[CR3DT - CC-3DT++ Tracker]]
- Related: 
	- [[BEVDet (2022)|BEVDet]]  
	- [[CenterPoint]] 
	- [[PointPillars]] 
	- [[Lift, Splat, Shoot]]