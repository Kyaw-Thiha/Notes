# MOSFET
[[MOSFET]] is an abbreviation for metal oxide [[semiconductor]] field effect transistors, and is used for switching or amplifying signals in electronic circuits.
![image|300](https://notes-media.kthiha.com/MOSFET/09a48be3e4ffc0dd66319e316d9f623b.png)

It operates by utilizing a gate terminal to create an electric field.
Hence, it can regulate conductivity between the drain and source, allowing it to act as an efficient electronic switch. 

---
## NMOS and PMOS

In N-channel([[#NMOS and PMOS|NMOS]]) and P-channel([[#NMOS and PMOS|PMOS]]), the gate voltage controls the current flow between the drain and the source.
![P and N-channel MOSFETs|300](https://res.cloudinary.com/rs-designspark-live/image/upload/c_limit,w_643/f_auto/v1/article/What_is_mosfet_7e47df2da83b4a21b7ab68272f549aadf2c9957f)
![P-MOS and N-MOS|300](https://www.powerelectronicsnews.com/wp-content/uploads/sites/3/2024/05/Fig1.jpg)

---
### NMOS
- **Operation**: Turns on when the gate voltage is higher than the source voltage $(V_{GS} > V_{th})$.
- **Performance**: Uses electrons, which have higher mobility.
  This lead to faster switching and lower on-resistance.
- **Application**: Used for high-speed switching and power management for better performance.

---
### PMOS
- **Operation**: Turns on when the gate voltage is lower than the source voltage $(V_{GS} < V_{th})$
- **Performance**: Uses holes, resulting in higher resistance.
  Require larger die size to achieve same $R_{DS(on)}$.
  Hence, they are less efficient.
- **Application**: Preferred for high-side switching to simplify driver circuitry.

---
## Power Consumption
Turning transistors on and off dissipates power and takes time.
- Dynamic power is the power used to charge capacitance (ability of a system to store charges), when signals change between $0$ and $1$.
- Static power is power that is used even if the system is idle (not changing between $0$s and $1$s).

---
## See Also
- [A Tutorial](https://www.electronics-tutorials.ws/transistor/tran_6.html)
- [[Electricity]]
- [[Semiconductor]]
- [[MOSFET]]
- [[Capacitor]]