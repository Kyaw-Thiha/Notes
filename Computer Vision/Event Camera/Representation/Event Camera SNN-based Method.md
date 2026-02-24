## Spiking Neural Network Method
`Spiking Neural Network (SNN)` process events asynchronously using discrete spike pulses (like biological neurons)

This incorporate time as an intrinsic part of the computational model rather than through explicit timestamp encoding.

**Motivation**
Events are spikes (discrete, asynchronous brightness changes).
`SNN` are the **native computational substrate** for event data, matching the biological mechanism that inspired event cameras.

**Advantages**
- **Native asynchronous processing** 
  Events drive neuron dynamics directly, no forced synchronization
- **Biological plausibility** 
  Matches neuromorphic hardware 
- **Potential energy efficiency** 
  Sparse spike communication (only active neurons communicate)

**Challenges**
- **Training Difficulty**
  Discontinuous `activation functions` means [[Backpropagation|standard backpropagation]] doesn't work.
  Hence, 
	- Long training times
	- High computational costs
	- Cannot leverage standard deep learning infrastructure
- **Infrastructure Gap**
  Lack of specialized hardware for efficient `SNN` execution
  Lack of mature algorithms and training frameworks

---
