# CR3DT - Camera-RADAR Fusion Strategy

Technical details of the two-stage Camera-RADAR fusion strategy in [[CR3DT (2024)|CR3DT]].

---

## BEV Grid Configuration
- **Range**: 51.2m × 51.2m around ego-vehicle
- **Resolution**: 0.8m per grid cell
- **Grid dimensions**: 128 × 128

---

## Camera Feature Processing

**Input**: $6$ RGB images ($256 \times 704$ resolution)

**Processing**:
1. [[ResNet]]-50 backbone extracts image features
2. [[Lift, Splat, Shoot]] view transformer projects to BEV
3. Features lifted into 64-channel pillars

**Output**: `(64×128×128)` tensor

---
## RADAR Feature Processing

**Input**: $5$ RADAR sensors with $5$ sweeps accumulated
- Current sweep + 4 previous sweeps
- RADAR has higher data rate than camera
- All sweeps within current frame timestamp 
  (alleviates sparsity)

**Feature Encoding** ([[PointPillars]]-inspired):
- **18 channels per point**:
  - Spatial: $(x, y, z)$ position
  - Velocity: $(v_{x}, v_{y}, v_{z})$
  - RADAR-specific: reflectivity, etc.
- **Aggregation**: Points averaged within each grid cell
- No augmentation needed (RADAR already info-rich vs LiDAR)
- Implicitly adds centroid of point cloud per pillar

**Output**: `(18×128×128)` tensor

---

## Two-Stage Fusion

### Stage 1: Initial Concatenation
Concatenate camera and RADAR features before `BEV encoding`:

```python
Camera BEV:  (64×128×128)
RADAR BEV:   (18×128×128)
             ──────────────
Concat:      (82×128×128)
	    ↓
ResNet BEV Encoder
        ↓
Encoded:     (256×128×128)
```

### Stage 2: Residual Connection
Add raw RADAR features back after encoding:

```python
Encoded:     (256×128×128)
RADAR (raw): (18×128×128)
             ──────────────
Concat:      (274×128×128)
             ↓
CenterPoint Detection Head
```

**Why residual connection?**
- Preserves raw RADAR features after BEV encoding
- Prevents information loss during encoding
- Critical for preserving velocity information

---

## Design Choice: Pillars vs Voxels

**Rejected: [[VoxelNet]]-style**
- 10 bins in z-dimension → `(82×10×128×128)`
- Requires 3D convolution "BEV compressor"
- Result: Lower performance

**Chosen: [[PointPillars]]-style**
- No z-discretization → direct pillar aggregation
- Simpler architecture, better performance
- **Insight**: Height less critical than lateral position + velocity for driving

---

## Key Insights

**Why this fusion strategy works**:
- Pillar aggregation simpler and more effective than voxel discretization
- Residual connection prevents velocity information bottleneck
- RADAR sweeps accumulation alleviates sparsity without violating online constraint
- Raw RADAR features remain accessible throughout the pipeline

---

## See Also
- [[CR3DT (2024)]] 
- [[BEVDet (2022)|BEVDet]] 
- [[PointPillars]] 
- [[VoxelNet]]