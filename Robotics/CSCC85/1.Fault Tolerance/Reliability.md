# Reliability
#robotics/fault-tolerance/reliability 

[[Reliability]] $R(t)$ is a measure of probability that the system will perform its function correct over an interval of time $[t_{0}, \ t]$.

![image|300](https://notes-media.kthiha.com/Reliability/8b270cf215b76389a75414473d53c1f8.png)

Mainly used for systems where 
- Short periods of incorrect behavior are unacceptable 
  E.g: Airplane flight control system
- Repairs are impossible
  E.g: Planetary exploration robots

---
### Measuring Reliability
We can use [[Reliability|success tree]] and [[Reliability|failure tree]] to measure the reliability of a system.

[[Reliability|Success Tree]] 
Logical model that shows how component-level successes combine to determine overall system success.

[[Reliability|Failure Tree]] 
Logical model that shows how component-level failures combine to determine overall system success.

---
`How to Measure`
The system is decomposed into different levels of components
- Basic Events: Component-level Failures/Successes
- Intermediate Events: Subsystem Outcomes
- Top Event: Overall outcome of the system

At each level, the component's outcomes probability can be represented using `logic gates` like:
- `OR Gate`
  E.g: Robot stops if motor fails OR battery fails
- `AND Gate`
  E.g: Localization fails only if lidar AND odometry fail
- `k-out-of-n Gate`
  E.g: 2-out-of-3 range sensors required

---
## Increasing Reliability
Use redundancy! 
But the reason why we don't overdo redundancy is because
- cost
- power
- space
- weight

For fault tolerance, we could do 
- redundancy of components
- redundancy of software
- RAID: Self-healing memory.

---
## Example

### Question
Suppose we have a system that requires 2 components to work: 
- A computer module with $0.98$ reliability.
- A power source with $0.93$ reliability.

### Naive Solution
The easiest solution could be formed by `AND Gate`.

$$
0.98 \times 0.93 = 0.91
$$

This shows that a collection of parts is less reliable than the individual parts alone.Solution-2

---
### Optimal Solution
Now, let's consider using redundancy in the components.

![image|300](https://notes-media.kthiha.com/Reliability/ac17bc00596acb7755c709d284dbe997.png)

$$
1-  \left( (1-0.93) \times (1-0.93) \right) = 0.995
$$
Using `OR Gate` makes the system more reliable than individual parts.

Implementing it to our system, we get

![image|300](https://notes-media.kthiha.com/Reliability/33e45865212e148007fc0a06d17b85a0.png)
which has better reliability than the original system.

---
### Alternative Solution

Consider this alternative solution.
![image|300](https://notes-media.kthiha.com/Reliability/0bb6d1318fb094caaded9b1b80ba8b06.png)
Note that this has better reliability than the original system, but has lower reliability than the optimal system.

---
## See Also
- [[Reliability]]
- [[Fault Tolerant Systems]]
- [[Fault Tolerances]]