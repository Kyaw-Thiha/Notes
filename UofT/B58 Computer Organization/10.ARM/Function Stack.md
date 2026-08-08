# Function Stack
![image|500](https://notes-media.kthiha.com/ARM-Stack/62874b20699cf7a9c4fb7159ef15a0af.png)

---
## Pushing/Popping the Stack
The stack grows downward such that the stack pointer `R13=SP` always point to next free address minus $4$.

![image|400](https://notes-media.kthiha.com/ARM-Stack/f20e62a0d96223bd3e44556fb59210c7.png)

`PUSH Rs` expands to
- R13 -= 4
- RAM[R13] = RS

`POP Rd` expands to 
- Rd = RAM[R13]
- R13 += 4

---
## Calling Convention

![image|300](https://notes-media.kthiha.com/ARM-Stack/159cf165b61470839b09b3d3ad226da5.png)

---
## Function Stack

![image|400](https://notes-media.kthiha.com/Memory-Map/d6c2a17e38878da3067e1c27db568aa0.png)

- `R0-R3`: The first four arguments in, return value out.
- `R4-R11`: Callee must preserve these. General Purpose.
- `R11`: Can be repurposed as frame/base pointer(`FP`). Marks the start of the current function's stack context.
- `R13`(`SP`): Stack pointer. Always point to the current top of the stack and decreases as things get pushed.
- `R14`(`LR`): Return address. Where to jump back to when the function finishes.

---
### Calling Sequence

![image|350](https://notes-media.kthiha.com/ARM-Stack/c9325238c18502c0b05a1224120c67c8.png)

---
## See Also
- [[ARM-32 Instruction Set]]