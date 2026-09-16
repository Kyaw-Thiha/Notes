# Hardware Fault Tolerance
Robotic systems are made up of sensors whose readings is essential for the robot's decision making.

The gold standard for achieving this is `Triple Module Redundancy` which uses 3 independent components to separately provide the desired function.
Think of 3 redundant batteries, or GNS+Lidar+IMU sensor-fusion.

Note that we choose $3$ modules instead of $2$ so that we can always break ties.

---
## Sensor Noise
Note that the readings from the sensors are going to be different due to sensor noise, physical location of sensor, and temporal misalignment.

Thats why we usually take median vote.

---
To decide how to use these readings, we can implement a `voting module`, which can use one of:
- Majority Vote
- Median Value
- Complicated Ruleset Function (like [[Neural Network]])

---
