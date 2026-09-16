# Information Fault Tolerance
The data inside a robotic system can be corrupted due to data transmission error, noise/electromagnetic interference in circuits, or failure of storage media.

---
## Information Consistency
Use multiple correlated sources of information.
Use all available sources of information: for example history.

But note that error accumulates meaning history would become unreliable over time.

---
## Techniques

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