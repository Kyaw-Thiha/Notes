## Voxel-based Representation
Map events into discretized **3D spatiotemporal grids** (`voxels`) where the third dimension is time.

### Key Methods
**Spatial-Temporal Voxel Grid**
- Linearly weighted accumulation when inserting events into temporal bins
- Improve temporal resolution beyond simple binning

**Time-Ordered Recent Event Volume**
- Preserve raw spike temporal information with minimal loss
- Compact encoding maintaining temporal ordering

---