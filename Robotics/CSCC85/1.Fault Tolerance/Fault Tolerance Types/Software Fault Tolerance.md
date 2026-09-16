# Software Fault Tolerance
Robotics software has to be able to detect and compensate for faults, handle error input, and be bug-free.

---
### Categorizing Failure Modes
Analyze and detail the specifications of the system, before listing and categorising the failure modes.

---
### N-Version Programming
Have $N$ different teams implement the system independently, then vote on the result.

Eg: Use in NASA space shuttles, airlines. They used $2$ versions.

---
### Testing
If teams cannot afford [[#N-Version Programming]], invest in heavy testing/certification. 
Eg: Airline, health care.

---
## Spec
Ensure we use fixed and extremely detailed spec.
Don't add last minute codes as is example in flight 737.

---
### Consistency Checking
Check for consistency of sensor readings, immediate results, and system outputs by
1. Forming a `model` to predict immediate future state, given current state and input readings.
2. Using `alternative sensors`
   E.g: Missing acceleration value can be computed from velocity readings
3. `Human-in-the-loop`
4. Add recovery blocks such that if $1^{st}\text{ software}$ does not pass an `acceptance block, use the $2^{nd}\text{ version}$

---