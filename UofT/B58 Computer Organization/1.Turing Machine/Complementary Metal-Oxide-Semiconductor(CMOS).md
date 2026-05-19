# CMOS
[[Complementary Metal-Oxide-Semiconductor(CMOS)|CMOS]] is a type of fabrication process that uses complementary and symmetrical pairs of [[MOSFET|p-type]] and [[MOSFET|n-type]] [[Transistor|MOSFET]] for logic functions.

---
## Motivation

Consider the following circuit diagrams for forming a `NOT gate`.

**Naive Implementation**
![image|150](https://notes-media.kthiha.com/Complementary-Metal-Oxide-Semiconductor(CMOS)/acba4381e04c5eae5c85431397973d96.png)
This wouldn't work since this will lead to short circuit.

**Adding in resistor**
![image|150](https://notes-media.kthiha.com/Complementary-Metal-Oxide-Semiconductor(CMOS)/c9769800bb45b2ea76ff9f56effd4aeb.png)
This would work, but adding in resistor meant wasting power.

**CMOS**
![image|300](https://notes-media.kthiha.com/Complementary-Metal-Oxide-Semiconductor(CMOS)/a1b17055566e1315e6bef0c8c907f7e6.png)
Adding in complementary [[MOSFET|PMOS]] and [[MOSFET|NMOS]] would make it work.
- [[MOSFET|PMOS]] is connected to the power
- [[MOSFET|NMOS]] is connected to the ground

---
## Building NAND gate
![|300](https://i.ytimg.com/vi/f3zRz0d9XA8/maxresdefault.jpg)
- Set up [[MOSFET|NMOS]] in series to act as an `AND gate`.
- Set up [[MOSFET|PMOS]] in parallel to act as an `OR gate`.z

---
### NAND Gate Implemenation
The following is the `NAND gate` implementation using [[Complementary Metal-Oxide-Semiconductor(CMOS)|CMOS]].
![image|350](https://notes-media.kthiha.com/Complementary-Metal-Oxide-Semiconductor(CMOS)/3e06051b2c546bb6b8da4438c62cefe3.png)

It would have the following logic gates.
![image|500](https://notes-media.kthiha.com/Complementary-Metal-Oxide-Semiconductor(CMOS)/1a5ab67ff5a55438fa89964ffadd6e31.png)

[[CMOS Example|More examples here]]

---
## See Also
- [[Transistor]]
- [[MOSFET]]
- [[CMOS Example]]
