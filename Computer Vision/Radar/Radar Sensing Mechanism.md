# Radar Sensing Mechanism
#radar/sensing-mechanism
`Radar sensing` converts reflected radio waves to digital signals.
Then, it extracts range, velocity, and angle using signal processing.

![Radar Sensing Mechanism|500](https://www.researchgate.net/profile/Manoj-Gofane/publication/314818626/figure/fig1/AS:471488911417344@1489423124390/Block-Diagram-of-RADAR-Transmitter-and-Receiver.png)

It always have at least $4$ main components:
- `Waveform Generator` 
- `TX Antennas` 
- `Signal Processors` 
- `RX Antennas` 

---
## Sensing Mechanism

`Transmitting`
1. `Waveform generator` 
   Creates a precisely controlled transmit signal.
2. `Power Amplifier` $(\text{PA})$
   Boost signal power before transmitting
3. `TX Antennas`
   Radiates the $\text{EM Energy}$ into space

`Target Interaction`
1. This transmitted wave reflects off objects depending on shape, material, orientation and [[Radar Cross Section (RCS)|RCS]].
2. The returned signal contains
	- Delay (`range`)
	- Frequency Shift (`Doppler Effect`)
	- Phase differences across antennas (`Angle`)

`Receiving`
1. `RX Antennas`
   Captures weak reflected signals.
2. `Low-Noise Amplifier` $(\text{LNA})$
   Amplify the received signal without adding noise.
3. `RF Mixer` $(\text{TX} \times \text{RX} \to \text{IF Signal})$
   Convert high-frequency RF signals into a usable low-frequency signal.
4. `IF signal conditioning` $(\text{Amplify + Filter})$
	- Remove unwanted frequency components
	- Match ADC input range
	- Improve SNR
5. `ADC` $(\text{Analog} \to \text{Digital})$
   Convert $IF$ signals into discrete-time samples
6. `Signal Processor`
	- $\text{Range FFT}$
	- $\text{Doppler FFT}$
	- $\text{Angle Estimation}$
	- $\text{Detection}$
	- $\text{Tracking}$

---
## See Also
- [[Radar]]
- [[Radar Cross Section (RCS)]]
