# Latches
[[SR Latch]] are a [[Sequential Circuit|feedback circuit]] with a more stable behaviour.

![image|300](https://notes-media.kthiha.com/Latches/1a90f73c5f3231fd01fd4db6a401eaa2.png)

---
## S'R' Latch with NAND Gate
![S'R' Latch|350](https://images.wevolver.com/eyJidWNrZXQiOiJ3ZXZvbHZlci1wcm9qZWN0LWltYWdlcyIsImtleSI6ImZyb2FsYS8xNzY0ODUxNzQ4NjI0LU5BTkQgQmFzZWQgU1IgTGF0Y2guanBnIiwiZWRpdHMiOnsicmVzaXplIjp7IndpZHRoIjo5NTAsImZpdCI6ImNvdmVyIn19fQ==)
The main thing to note is that the `NAND gate` would fire once of the input is $0$. 

The interesting thing is when $S'=1$ and $R'=1$.
- Since both inputs are $1$, they both can no longer activate the `NAND Gate`.
- Instead the prior values of $Q$ and $Q'$ makes the decision.
- Hence, $S'$ and $R'$ are called `set` and `reset` respectively.
	- When $S'=0, \ R'=1$, $Q$ is $1$.
	- When $S'=1, \ R'=0$, $Q$ is $0$.
	- When $S'=1, \ R'=1$, $Q$ is same as previous state.

For going from $00$ to $11$,
- It depends on if it changes from $00 \to 01 \to 11$ or $00 \to 10 \to 11$
- This means it has a race condition.
- Hence, it is unstable.

Therefore, 
- $00$ is considered a `forbidden state` in [[SR Latch|NAND-based S'R' Latches]].
- $11$ is considered a `forbidden state` in [[SR Latch|NOR-based SR Latches]].

---
## SR Latch with NOR Gates
![SR Latch|350](https://i.sstatic.net/zk3PI.jpg)

---
## Propagation Delay
The output signals don't change instantaneously, but with a certain delay.

![SR Latch Propagation Delay|250](https://ranger.uta.edu/~carroll/cse2341/summer99/html%20files/chapter_6/img016.GIF)

---
