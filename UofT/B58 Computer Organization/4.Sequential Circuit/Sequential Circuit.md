# Sequential Circuit
A [[Sequential Circuit|sequential circuit]] is a circuit whose internal state can change over time, where the same input values can result in different output values.

![|300](https://i.ytimg.com/vi/fLN1YOmuAr8/sddefault.jpg)

---
## Difference between combinational and sequential circuits
- `Combinational circuits` are circuits where output values are entirely dependant on input values.
- [[Sequential Circuit|Sequential circuits]] are circuits that depends on both input values and previous state of circuit.

---
## Examples
### AND Gate with Feedback
![image|350](https://notes-media.kthiha.com/Sequential-Circuit/118dde3a1e4360d4ea4c2ede3167fe08.png)

Note that $(0,1,0)$ is a transcient state since it will become $(0,0,0)$ immediately. Hence once the output is $0$, the circuit will be stuck at $0$ no matter how $A$ is changed.

---
### NAND Gate with Feedback
![image|350](https://notes-media.kthiha.com/Sequential-Circuit/24dc38981097256ea65c92f941bbe64e.png)

And its waveform behaviour is
![image|350](https://notes-media.kthiha.com/Sequential-Circuit/bb318dd98133e4085d6359e45cea2e0a.png)

---
### NOR Gate with Feedback
![image|350](https://notes-media.kthiha.com/Sequential-Circuit/577e462ff93d477a30bd2633f41831d0.png)

---
## Feedback Behaviour
Unlike the `AND` and `OR` gates, in `NAND` and `NOR gates` the output $Q_{t+1}$ can be changed based on $A$ instead of getting stuck.

However, these feedback gates can enter an unsteady state.

---
## Design Procedure of Combinational and Sequential Circuit
In `combinational circuits`, we
- first state the desired behaviour
- draw the truth table
- logic expression
- build circuit

In [[Sequential Circuit|sequential circuits]], we
- first state the desired behavior
- draw the [[Finite State Machine|finite state machine]]
- build circuit with [[Flip-Flop|flip-flops]]

---
## See Also
- [Video about Sequential Circuits](https://youtu.be/fLN1YOmuAr8?si=yKbWlE-fHf_6yTof)
- [Lecture Video covering sequential circuits](https://youtu.be/baFa6_Wv7cg?si=cx0sgXi6A3kkfwGR)
