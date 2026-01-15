# Radar Waveform Types
#radar/waveforms
Radar systems can be classified based on how they generate & modulate transmitted waveforms.

![Radar Waveform Types|300](https://ars.els-cdn.com/content/image/3-s2.0-B9780128225486000832-f00083-02-9780128225486.jpg)

---
## Generation Classification
Radar systems can be categorized based on how they generate the radiated waveform.
- `Continuous-Wave (CW) Radars`: Radiated waveform is continuous
- `Pulse Radar`: Signal is emitted intermittently a over short duration

---
## Modulation Classification
Radar systems can be categorized based on how information is embedded $(\text{modulated})$ onto carrier waveform.

> Note that `modulation` determines what physical property of the signal is varied to encode range, velocity, and angle.

---
`Unmodulated Radar`
- Carrier signal has constant frequency & phase.
- Can only reliably measure presence or radial velocity $(\text{via Doppler})$.
- Mostly of historical or niche usage

---
`Amplitude-Modulated (AM) Radar`
- Carrier signal has varying amplitude to encode information.
- Highly sensitive to noise & attentuation.
- Poor robustness means it is obsolete.

---
`Frequency-Modulated Radar`
Information is encoded by varying `instantaneous frequency`.

`Frequency-Modulated Continuous Wave` $(\text{FMCW})$
This is the main form of frequency-modulated radar.
- Frequency changes over time, producing a `chirp waveform`
- Range extracted from `frequency difference`
- Excellent `range-velocity resolution`
  Dominant in automotive & short-range sensing

[[FMCW|Read More]]

---
`Phase-Modulated Radar`
Information is encoded by varying `signal phase`.

`Phase-Modulated Continuous Wave` $(\text{PMCW})$
- Phase modulated using digital codes.
- Range obtained via `code correlation`.
- Higher range accuracy than $\text{FMCW}$.
  Increasing interest in modern radar research.

[[PMCW|Read More]]

---
## System Architecture Classification
Radar systems can be categorized based on their architecture.

`Scanning Radar`
- Consists of a physically rotating radar sensor.
- Provides highly accurate polar image of the surroundings by measuring target distances at various angle.
- Target distances are obtained by performing `FFT` on ADC samples for each angle, and identifying the peaks.

`System-on-Chip (SoC) Radar`
- Integrates processing units into limited no. of chips
- Demonstrate lower weight and power consumption 
- Accuracy & precision depends on antenna array & processing

> $\text{Scanning Radar}$: `Mechanically` scan space.
> $\text{SoC Radar}$: `Electronically` senses space using antenna arrays + on-chip processing.

---
## See Also
- [Main Reference Paper](https://arxiv.org/abs/2410.19872)