# Main Memory
The [[Main Memory|main memory]], like [[Register|register files]], is made up of a decoder and rows of memory unit.

![image|300](https://notes-media.kthiha.com/Main-Memory/501883252d6558fc7c5f5c80a8e1e5c9.png)

---
### Computer Memory Hierarchy
As sorted by data access speed:
- [[Register|Registers]]: in the processor
- [[Cache]]: several levels, the closest is next to processor
- [[Main Memory|Memory]]: off-chip
- Hard Disk: for virtualisation; require OS support to access
- Network 

---
### Memory vs Register
- [[Main Memory|Memory]] houses most data values being used by program.
- [[Register|Registers]] are more local data stores, meant to be used to execute an instruction.
	- [[Register|Registers]] are not meant to host memory between instructions.
	- Exception is the [[Register|stack pointer register]], which is sometimes in the same register file as the others.

---
### One-Hot Decoder
The decoder takes in the m-bit binary address and activates a single row in the memory array.

![image|300](https://notes-media.kthiha.com/Main-Memory/d6c601fd9cbaa3b255c4928e731d09a9.png)

---
### Controlling the flow
Since some lines([[Data Bus|buses]]) will now be used for both input and output, we use a [[Tri-State Buffer|tri-state buffer]].

![image|150](https://notes-media.kthiha.com/Main-Memory/39ec0c77f7afd8229dce9587334745f4.png)

When `WE`(`write enable`) signal is low, [[Tri-State Buffer|buffer output]] is high impedence signal (connected to neither high voltage nor ground).

We can control `c0`,`c1`,`c2` so that only one of the devices output is written to the [[Data Bus|bus]].
![image|250](https://notes-media.kthiha.com/Main-Memory/28f5dc2f481f9621e799c4589b2b9b99.png)

In general, the [[Data Bus|bus]] can be read by multiple devices at the same time, but can only be written by one device at a time.

---
## RAM Storage Cells
For storing a single bit, each row is made of an array of storage cells.

There are multiple ways of representing the cells such as the `RAM cell`(basically a [[SR Latch|gated latch]]).

![image|250](https://notes-media.kthiha.com/Main-Memory/857633f5d68e50557f743cf00fb57ace.png)

### RAM Slice Model
The `word select` signals determine which row to send out on the `C lines`.
![image|350](https://notes-media.kthiha.com/Main-Memory/5d7f1f48340050c11ee6128e528f56bb.png)

---
## Amdahl's Law
Most processor spends most of their time waiting.
No matter how fast we makes a processor, if memory is too far away, we'll just spend more time waiting.

As a result, [[Amdahl's Law]] tells us that memory access is an aspect of performance that has become increasingly important.

---
