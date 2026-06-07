## Clock Signals
[[Clocked SR Latch|Clocks]] are regular pulse signals where the high value indicates that the output of the latch maybe sampled.
![Clock Signal|300](https://hades.mech.northwestern.edu/images/c/cd/D_flipflop_timing.gif)

### Signal Restrictions
The following restrict how fast the [[Clocked SR Latch|latch circuit]] can be sampled:
- latency time of [[Transistor|transistors]]
- setup time for [[Clocked SR Latch|clock signal]]
	- jitter
	- Gibbs phenomenon

It can be measured by `frequency` which is the number of pulses occurred per second.

---
## Clocked SR Latch
![image|250](https://notes-media.kthiha.com/Clocked-SR-Latch/b7f6a833cfea87e5ea5e86c89b9c8322.png)
We can get the [[Clocked SR Latch]] by adding another layer of `NAND Gate` to the [[SR Latch|S'R' Latch]].

The [[Clocked SR Latch|clock]] $C$ is connected to a pulse signal that alternates regularly between $0$ and $1$.

---
### When clock is HIGH 
If the [[Clocked SR Latch|clock]] is `HIGH`, 
- the first `NAND Gates` invert the inputs of $S$ and $R$.
- this get inverted again in the output.
- 
![image|200](https://notes-media.kthiha.com/Clocked-SR-Latch/c2f3a0db947b490a9301240cf5b5f79e.png)

Setting both inputs to $0$ will maintain the prior output values.

---
### When clock is LOW
If the clock is `LOW`, the low clock input prevents the change in $S$ and $R$ from reaching the second stage of `NAND gates`.
![image|200](https://notes-media.kthiha.com/Clocked-SR-Latch/f90f6239a5b37b6624c018ff59c3fb52.png)
> [[Clocked SR Latch|Clock]] needs to be high in order for the inputs to have any effort.

---
### Symbology
![image|300](https://notes-media.kthiha.com/Clocked-SR-Latch/876f9acad33738dd263cc8ff7216547e.png)
Note that the small NOT circle after $\bar{Q}$ output is not an extra `NOT gate`; it's just a notation to denote inverted output value.

---
### Truth Table
![image|200](https://notes-media.kthiha.com/Clocked-SR-Latch/b9ec630584d1285624b28d5d041dc179.png)

---
