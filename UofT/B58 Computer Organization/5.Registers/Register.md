# Register
In computer architecture, [[Register|registers]] are the CPU's most local storages, an are the lowest level of memory hierarchy.

![|350](https://diveintosystems.org/book/C11-MemHierarchy/_images/MemoryHierarchy.png)

---
## Shift Register
[[Register|Shift registers]] are a [[Sequential Circuit|digital circuit]] made of cascading [[Flip-Flop|flip-flops]] that temporarily store and move binary data.
![image|350](https://notes-media.kthiha.com/Shift-Register/a7af526248ce63a94dabd02adbc75322.png)

A series of [[Flip-Flop|D flip-flops]] can store a multi-bit value $(\text{E.g: 16-bit int})$.
Data can be shifted into this register one bit at a time $(\text{over 16 clock cycles for 16-bit integer})$.

---
## Load Register
[[Register|Load register]] can load a [[Register|register]]'s value all at once, by feeding signals into each [[Flip-Flop|flip-flops]].
![image|250](https://notes-media.kthiha.com/Register/262d1511f71b7f0674e4b0820724004a.png)
In this $\text{4-bit}$ [[Register|load register]], $4$ bits can be stored in one clock pulse.

### Enable
To control when this [[Register|register]] is allowed to load its values, we introduce the [[Flip-Flop|D flip-flop with enable]]:
![image|350](https://notes-media.kthiha.com/Register/296b48754b48140256c6a7f9c0a8dde8.png)

Implementing the [[Register|register]] with these [[Flip-Flop|special D flip-flops]] will maintain values in the register, until overwritten by setting EN to high.
![image|350](https://notes-media.kthiha.com/Register/883caf6bf38d9c03c507bd08d1ff8396.png)

---
