## Image-Based Representation
Convert event streams into 2D image representations. 
This allows leveraging existing [[Convolutional Neural Network (CNN)|CNN architectures]] designed for traditional frame-based cameras.

**Polarity-Based Stacking**
- Separate channels for positive/negative events
- Histogram evaluation per polarity $\to$ $2\text{-channel  representation}$
- Merge into synchronous frames

**Timestamp-Based Stacking**
- Encodes temporal information alongside event counts
- Preserves both **when** events occurred and **how many**
- Channels typically encode: polarities, timestamps, event counts

**Event Count Based Stacking**
- Fixed number of events per frame (not fixed time window)
- Addresses asynchronous nature 
  Events don't trigger uniformly in time
- $\text{Constant-}N\text{ sampling}$ instead of $\text{Constant-}\Delta t \text{ sampling}$

**Combined Polarity+Timestamp Stacking**
- `EV-gait`: $4$-channel frame representation
	- $2$ channels: positive/negative polarities
	- $2$ channels: temporal characteristics
- `Adaptive Exposure Control`
	- Dynamic exposure time control
	- Adaptive inter-slice time interval adjustment
	- **Motivation** 
	  Optimize slice feature quality across varying motion speeds and scene structures
	- Ensures robustness in dynamic scenes 
	  (fast-moving objects vs. static scenes require different temporal windows)
