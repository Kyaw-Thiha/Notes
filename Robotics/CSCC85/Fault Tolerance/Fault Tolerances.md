# Fault Tolerances
#robotics/fault-tolerance 

`Fault tolerance` is the property of a system that allows it to handle a certain number of flaws while
still fulfilling its correct function. 

There are 3 different faults to consider:
1. Hardware Fault Tolerance
2. Software Fault Tolerance
3. Information Fault Tolerance

---
## Hardware Fault Tolerance
Robotic systems are made up of sensors whose readings is essential for the robot's decision making.

The gold standard for achieving this is `Triple Module Redundancy` which uses 3 independent components to separately provide the desired function.
Think of 3 redundant batteries, or GNS+Lidar+IMU sensor-fusion.

Note that the readings from the sensors are going to be different due to sensor noise, physical location of sensor, and temporal misalignment.

To decide how to use these readings, we can implement a `voting module`, which can use one of:
- Majority Vote
- Median Value
- Complicated Ruleset Function (like [[Neural Network]])

---
## Software Fault Tolerance
Robotics software has to be able to detect and compensate for faults, handle errornous input, and be bug-free.

`Categorizing Failure Modes`
Analyze and detail the specifications of the system, before listing and categorising the failure modes.

`N-Version Programming`
Have $N$ different teams implement the system independantly, then vote on the result.

`Consistency Checking`
Check for consistency of sensor readings, immediate results, and system outputs by
1. Forming a `model` to predict immediate future state, given current state and input readings.
2. Using `alternative sensors`
   E.g: Missing acceleration value can be computed from velocity readings
3. `Human-in-the-loop`
4. Add recovery blocks such that if $1^{st}\text{ software}$ does not pass an `acceptance block, use the $2^{nd}\text{ version}$

---
## Information Fault Tolerance
The data inside a robotic system can be corrupted due to data transmission error, noise/electromagnetic interference in circuits, or failure of storage media.

- `Error Detection`
Use techniques like `Parity Checking` and `Error Correcting Checksums`.

- `Data Redundancy`
Compare values from multiple input, and store data in multiple hardwares.

- `Checkpointing`
Create periodic checkpoints where data, outputs and state of the system are known to be correct.
Thus, if error is detected, we can roll-back to nearest checkpoint, and reconstruct the correct sequence of processing.

---
### Voting-out-configuration
For both `Triple Module Redundancy` and `N-Version Programming`, voting the module to use means that the faulty module will still have its input readings affecting future decisions.

We can solve this by voting out the module(s) that does not pass the `acceptance block`.

---
## See Also
- [[Fault Tolerant Systems]]
- [[Reliability]]