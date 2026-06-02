# Fault Tolerant Systems
#robotics/fault-tolerance 

In `Robotics` when performing critical tasks, systems cannot be allowed to fail due component failures or software bugs.

![Fault Tolerant Systems|500](https://moonlight-paper-snapshot.s3.ap-northeast-2.amazonaws.com/arxiv/fault-tolerant-multi-robot-coordination-with-limited-sensing-within-confined-environments-1.png)

---
### Measures

`1. Reliability` 
Reliability $R(t)$ is a measure of probability that the system will perform its function correct over an interval of time $[t_{0}, \ t]$.

Mainly used for systems where 
- Short periods of incorrect behavior are unacceptable 
  E.g: Airplane flight control system
- Repairs are impossible
  E.g: Planetary exploration robots
[[Reliability|Read More]]
  
`2. Availability`
Availability $A(t)$ is a measure of probability that a system is performing correctly and is available to carry out its function at time $t$.

`3. Maintainability`
Maintainability is a measure of probability that the system can be restored to correct operation within time bound after failure.
- `MTTR`: Average time to restore service after failure
- `MTTF`: Average uptime until failure

`4. Safety`
Safety is a specific measure of probability that the system exhibits specific dangerous behaviours.

---
## Fault Tolerance
Fault tolerance is a property of a system that allows it to handle a certain number of flaws while still fulfilling its correct function.

`Hardware Fault Tolerance`
We can handle failures in hardware components by:

- Use `Triple Modular Redundancy (TMR)` by using multiple sensors
- Use redundancy (TMR) on components like batteries

`Software Fault Tolerance`
The software needs to be free of bugs, while also able to detect and compensate for faults. We handle this by:
- List and categorize possible failure modes, causes, and severity
- Use `N-version programming`
- Check for consistency
- Use redundancy blocks

`Information Fault Tolerance`
The data inside the system can be corrupted when being transferred or stored. We handle this by:
- Detecting error using techniques like `parity checking` and `error correcting codes`
- Data duplication
- Checkpoint

[[Fault Tolerances|Read More]]

---
## See Also 
- [Paco's Notes](https://www.cs.utoronto.ca/~strider/docs/C85_FaultTolerance.pdf)
- [[Fault Tolerances]]
- [[Reliability]]