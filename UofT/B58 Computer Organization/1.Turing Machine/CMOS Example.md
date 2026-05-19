# CMOS Example

> Suppose we want to build a circuit for `NOR gate`.

## Non-Complementary 
Note that when building with [[MOSFET|NMOS]] or [[MOSFET|PMOS]] only, we put the resistor on the side which we want to be don't want to be on by default.
- If we put resistor on power, and if both power and ground are on, ground will be the output.
- If we put resistor on ground, and if both power and ground are on, power will be the output.

---
### NMOS Only
Build `NOR gate` using only [[MOSFET|NMOS]] only.
![image|200](https://notes-media.kthiha.com/CMOS-Example/391f70bbb1f34fd29174610345db7591.png)
- A and B are in parallel.
- Resistor is on the side of the power.

---
### PMOS Only
![image|200](https://notes-media.kthiha.com/CMOS-Example/7926ff8045bb6fadd307133c19ac27f1.png)
- A and B are in series.
- Resistor is on the side of the ground.

---
## Complementary
- From a logic input, 
	- if [[MOSFET|PMOS]] is connected to power, [[MOSFET|NMOS]] should be connected to the ground (`NOR Gate`)
	- if [[MOSFET|PMOS]] is connected to ground, [[MOSFET|NMOS]] should be connected to the power (`NAND Gate`)
- Given two logic inputs,
	- `NOR Gate`: [[MOSFET|PMOS]] are in series and [[MOSFET|NMOS]] are in parallel 
	- `NAND Gate`: [[MOSFET|PMOS]] are in parallel and [[MOSFET|NMOS]] are in series 

---
### NOR gate example
![image|200](https://notes-media.kthiha.com/CMOS-Example/a8f0fc78d239b8cedf028513a45bdc04.png)

---
### NAND Gate
![image|350](https://notes-media.kthiha.com/Complementary-Metal-Oxide-Semiconductor(CMOS)/3e06051b2c546bb6b8da4438c62cefe3.png)

---
## See Also
- [[Complementary Metal-Oxide-Semiconductor(CMOS)]]
- [[Transistor]]
- [[MOSFET]]