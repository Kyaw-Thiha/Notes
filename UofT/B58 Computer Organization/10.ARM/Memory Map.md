# Memory Map
The [[Function Stack|stack]] and [[Heap|heap]] are both dynamic but they grow towards each other from opposite ends.

![image|300](https://notes-media.kthiha.com/Memory-Map/808014ce8f8d3790a06a8bbb5be74855.png)

- `Static/Global Variables` get one fixed address forever.
- In [[Function Stack|Stack Memory]], every function call gets its own fresh frame. So recursive calls each get their own locals.
- [[Heap]] is dynamic memory which is hybrid.
	- Like static, it survives after function returns.
	- Like stack, it can be created/destroyed on demand.
- `Memory-mapped I/O` sits at their special addresses.
  Reading/writing those registers don't touch RAM. Instead, it talks to hardware peripheral.

---
## Function Stack

![image|400](https://notes-media.kthiha.com/Memory-Map/d6c2a17e38878da3067e1c27db568aa0.png)

- `R0-R3`: The first four arguments in, return value out.
- `R4-R11`: Callee must preserve these. General Purpose.
- `R11`: Can be repurposed as frame/base pointer(`FP`). Marks the start of the current function's stack context.
- `R13`(`SP`): Stack pointer. Always point to the current top of the stack and decreases as things get pushed.
- `R14`(`LR`): Return address. Where to jump back to when the function finishes.

[[Function Stack|Read More]]

---
## Heap
![image|450](https://notes-media.kthiha.com/Memory-Map/a59ad569939d251740cab96340d400e4.png)

[[Heap|Read More]]

---
## Memory-mapped I/O & GIO
Instead of Syscalls, we can talk directly to hardware by reading/writing specific memory addresses that are wired to peripheral registers instead of RAM.

---
