## D-Latch
Since $S=1, \ R=1$ leads to forbidden state, we should try to prevent both of them from going HIGH.
![image|300](https://notes-media.kthiha.com/Clocked-SR-Latch/ee50f9801c06f77e1b2667d1dd8bdff7.png)

By making the inputs to $R$ and $S$ dependent on a single signal $D$, we avoid the indeterminate state system.

The value of $D$ now set output $Q$ as `LOW` or `HIGH`.
![image|200](https://notes-media.kthiha.com/D-Latch/0b282c0338da5622d60b5f90a7c18100.png)

---
### Latch Timing Issues
When the [[Clocked SR Latch|clock signal]] is `HIGH`, the output looks like this.
![image|300](https://notes-media.kthiha.com/Clocked-SR-Latch/0e7d16f0ab53fde622c49ef2a5f02774.png)
However, we want to have output changing only once when the [[Clocked SR Latch|clock pulse]] change.

---
## SR Flip-Flops
The solution is to 
- create a disconnect between circuit output and circuit input
- in order to prevent unwanted feedback and changes to output
- 
 ![image|350](https://notes-media.kthiha.com/D-Latch/8f1867d8354c9fd2c0564ba1cd43a038.png)
 Only one latch is active at one time.
 So in order for a change to propagate from input to output, we need to wait at least one clock cycle of both being active.

---
## Edge-Triggered D-Latch
Note that the [[#SR Flip-Flops]] still have the issues of unstable behaviour due to the forbidden state.

To solve this, we connect the [[Flip-Flop]] to the input of a [[SR Latch]].
![image|300](https://notes-media.kthiha.com/D-Latch/6a2ddde0e27dc9910d4d2bdc958fe996.png)
Hence, negative-edge triggered flip-flop like [[Flip-Flop|SR Flip Flop]].

---
### Flip-Flop Behaviour
- If the clock signal is `HIGH`, the input to the first flip-flop is sent out to the second.
- The second flip-flop doesn't do anything till the [[Clocked SR Latch|clock signal]] goes down again.
- When the clock goes from `HIGH` to `LOW`, 
	- the first flip-flop stops transmitting signal
	- and the second one starts

![image|400](https://notes-media.kthiha.com/D-Latch/42d0c9b5853dc3169397e22f02e23ca9.png)

- If the input to $D$ changes, the change isn't transmitted to the second flip-flop until the clock goes `HIGH` again.
- Once the clock goes `HIGH`, the first flip-flop starts transmitting at the same time that the second flip-flop stops.

![image|400](https://notes-media.kthiha.com/D-Latch/a9ddf4bb60219f6ab47a27f80ee8fca4.png)

---
### Notation
![image|300](https://notes-media.kthiha.com/D-Latch/b73b0859740834d0664c75f00a83ded6.png)

---