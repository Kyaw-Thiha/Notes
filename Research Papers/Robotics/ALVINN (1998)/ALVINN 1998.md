# ALVINN 1998
Link: https://proceedings.neurips.cc/paper/1988/file/812b4ba287f5ee0bc9d43bbf5bbe87fb-Paper.pdf

This is the first paper that uses neural network for autonomous navigation of a car.

![ALVINN](https://youtu.be/IaoIqVMd6tc?si=Jm4eWEbNJMUP5TRG)

## Architecture
`ALVINN` is a 3-layer neural network that takes inputs from camera & range sensor, and output direction of travel.

![ALVINN](https://i-blog.csdnimg.cn/blog_migrate/129889f4cafa9ca3b9418d40a1fc6df9.png)

First input layer contains 1217 neurons involving:
- `Retina-1`: represent camera input using $30 \times 32$ neurons
- `Retina-2`: represent range finder input using $8 \times 32$ neurons
- `Intensity Feedback Unit`: Single neuron that estimates whether whether the road is lighter or darker than the non-road in the previous image

The second layer is a hidden layer of 29 neurons.

The third output layer involves
- `Direction`: the first 45 units representing the curvature
- `Intensity Feedback Unit`: whether the road is lighter or darker than the non-road in the current image

In the direction component, 
- middle neurons represent `Travel Straight`
- neurons to the left represent `Left Turn` intensity
- neurons to the right neurons represent `Right Turn` intensity

Specifically, the nine units centered around the correct tum curvature unit are 0.10, 0.32, 0.61, 0.89,
1.00,0.89,0.61,0.32 and 0.10. 

### Jordan Network
Note that the road intensity compared to background is passed on from the previous frame to the next.

This representation through the neuron is called a `Jordan-style Feedback`, and is considered as one the first form of `recurrent neural networks (RNN)`.

It gives model access to one-step memory of its last decision.

`Jordan models` would get replaced by other 
- `early RNNs` in the 1980s-1990s, which get replaced by 
- `LSTM` in 1997, which get replaced by
- `GRU` (Simplfied LSTM) in 2014 which get replaced by
- `Transformer Models` in 2017

## Training
The model was trained using simulated images from a road simulator that outputs both camera image & range finder sensor output.
The model was trained on 40 epochs on 1200 simulated road snapshots.

## Lessons for Future
Since this is an ancient scroll of wisdom with small relevance on modern papers, the main key takeaways are
- **Data Diversity matters**
  For robustness you need varied conditions and also examples of how to recover from mistakes
- **Output as a distribution**
  They don’t output one steering value; they place a “hill” of activation over discrete curvature bins and pick the argmax
- **Sensor fusion via learning**
  Fuse multiple sensors and let the network learn the weighting.
- **Data > hand-coding**
  Let the data decide which features matter; don’t over-engineer feature pipelines.
- **Close the sim-to-real loop with field tests**
  Even early on, they validated on a real vehicle (NAVLAB). 
  Proving-in the loop beats only offline metrics.