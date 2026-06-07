## Sequential Circuit
Recall that [[Sequential Circuit|sequential circuits]] are circuits with an internal state.

To implement this internal memory, we use a form of [[Sequential Circuit|feedback]] from the output into the output.

[[Sequential Circuit|Read More]]

---
## SR Latch
[[SR Latch]] are [[Sequential Circuit|feedback circuits]] with a relatively stable behaviour.

They can be implemented with `NAND Gates`
![S'R' Latch|350](https://images.wevolver.com/eyJidWNrZXQiOiJ3ZXZvbHZlci1wcm9qZWN0LWltYWdlcyIsImtleSI6ImZyb2FsYS8xNzY0ODUxNzQ4NjI0LU5BTkQgQmFzZWQgU1IgTGF0Y2guanBnIiwiZWRpdHMiOnsicmVzaXplIjp7IndpZHRoIjo5NTAsImZpdCI6ImNvdmVyIn19fQ==)

or with `NOR Gates`.
![SR Latch|350](https://i.sstatic.net/zk3PI.jpg)

[[SR Latch|Read More]]

---
## Using Clock Signal
In order to sample the latches at regular frequency, we can connect a [[Clocked SR Latch|clock signal]] to our [[SR Latch]].
![image|250](https://notes-media.kthiha.com/Clocked-SR-Latch/b7f6a833cfea87e5ea5e86c89b9c8322.png)

[[Clocked SR Latch|Read More]]

---
## D Latch
To restrict against the forbidden state of $S=1,\ R=1$, we can restrict them to a single $D$ instead.
![image|300](https://notes-media.kthiha.com/Clocked-SR-Latch/ee50f9801c06f77e1b2667d1dd8bdff7.png)

## SR Flip-Flop
However, to make sure the output change only when the [[Clocked SR Latch|clock signal]] change, we can create a disconnect between circuit input and circuit output.
 ![image|350](https://notes-media.kthiha.com/D-Latch/8f1867d8354c9fd2c0564ba1cd43a038.png)

## Edge-Triggered D-Latch
Now to prevent forbidden state again, we reintroduce $D$ as
![image|300](https://notes-media.kthiha.com/D-Latch/6a2ddde0e27dc9910d4d2bdc958fe996.png)

[[D Latch|Read More]]

---
## Notation
![image|350](https://notes-media.kthiha.com/Latches-and-Flip-Flops/67caff358a3ee46fa5685033265bf0b7.png)

---
## See Also
- [Youtube Link](https://youtu.be/baFa6_Wv7cg?si=S1fcAbipsdNSWafm)
- [[Sequential Circuit]]
- [[SR Latch]]
- [[Clocked SR Latch]]
- [[D Latch]]