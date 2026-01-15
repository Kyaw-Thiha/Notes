# Phase-Modulated Continuous-Wave
#radar/waveforms/fmcw
`PMCW radar` measures distance by looking at time delay of a digital code using correlation.

![PMCW Radar|300](https://img1.wsimg.com/isteam/ip/84f61497-f6b2-4918-b1c2-c764c2788d10/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-02-16%20021416-11f9ba4.png/:/cr=t:0%25,l:0%25,w:100%25,h:100%25/rs=w:600,cg:true)

---
## General Overview
- A `single-frequency` signal is employed as the carrier wave at the transmission end.
- This carrier wave undergoes phase modulation using `encoding` methods $\text{(such as binary encoding)}$.
- This encoded carrier wave is then transmitted, received and processed to extract $\text{target distance}$, $\text{velocity}$ and $\text{angle information}$.

---
### Comparism to FMCW
[[FMCW]] uses `frequency difference` between transmitted and received signals for range detection.
> `PMCW radars` directly employs `digital code correlation`.

This drops the requirement of strict linearity in frequency ramping over time.


---
## See Also
- [Article about PMCW](https://radarsimx.com/2019/05/24/pmcw-radar/)
