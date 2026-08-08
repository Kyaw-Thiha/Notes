# UART
[[UART|UART (Universal Asynchronous Receiver Transmitter)]] sends data one bit at a time over a single wire.

![image|500](https://notes-media.kthiha.com/UART/6f28b901f23b2fa1706ff696e6da6688.png)

It does not use shared clock. Instead, timing is agreed in advance via a baud rate.

---
## Frame Anatomy
- At **idle**, the line is high(`1`).
- The **start bit** drops it to `0` for one bit period.
- **Data bits** are transmitted, with least-significant bits first.
- One or more **stop bits** bring the line back to `1`.

---
