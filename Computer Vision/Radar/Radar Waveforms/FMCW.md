# Frequency-Modulated Continuous-Wave
#radar/waveforms/fmcw
`FMCW` is a radar technique where a continuous signal whose frequency changes over time, is transmitted.

![FMCW|400](https://i.ytimg.com/vi/xUGWHGjCtII/hq720.jpg?sqp=-oaymwEhCK4FEIIDSFryq4qpAxMIARUAAAAAGAElAADIQj0AgKJD&rs=AOn4CLAqWTUYbEEjEOuW7sfDj69jEvdfhA)

`FMCW radar` measures distance by looking at frequency differences caused by a chirp $(\text{frequency ramp})$.

---
## FMCW Data Structure
After processing, `FMCW radar` forms a data cube of
$$
\text{Range}
\times \text{Doppler}
\times \text{Angle}
$$
This is also called `Range-Azimuth-Doppler` tensors.

`4D Radar`
Modern `MIMO FMCW radars` measures a signal of the form
$$
s(r, \theta, \phi, v)
$$
where
- $r$ is the `range`
- $\theta$ is the `elevation` $(\text{Up/Down Angle})$
- $\phi$ is the `azimuth` $(\text{Left/Right Angle})$
- $v$ is the `radial velocity` $(\text{Doppler})$

They achieve this stacking radars vertically.

---
## Modulation Mechanism

- An `FMCW radar` sends linear frequency ramps (chirps)
- A reflected chirp comes back time-delayed
- By mixing $\text{TX}$ and $\text{RX}$, the delay becomes a $\text{beat frequency (IF)}$.
  This gives `range`.
- Repeating chirps lets phase changes over time reveal `velocity`.
- Using multiple antennas, phase differences across space reveal `angle`.

---
### Terminologies
- `Frame`: Short period within which chirps are sent $\text{back-to-back}$.
- `Fast Time`: Duration of one chirp.
- `Slow Time`: Duration across multiple chirps.

---
### 1. Transmitting Signal
An `FMCW waveform`$(\text{a chirp})$ is a continuous wave signal whose frequency increases linearly
$$
f(t) = f_{c} + St
$$
where
- $f_{c}$ is the carrier frequency
- $S = \frac{B}{T_{c}}$ is the chirp slope
- $B$ is the bandwidth
- $T_{c}$ is the chirp duration $(\text{fast time})$

---
### 2. Range Esimation

`Reflection`
A target at range $R$ causes a round-trip delay:
$$
\tau = \frac{2R}{c}
$$
between the transmitted and received chirps.

> This delay can be estimated by mixing two chirps, and measuring the $\text{IF}$ of the mixed signal.

---
`Mixing`
Mixing the two chirps,
$$
\begin{align}
s_{IF} (t) = s_{rx}(t) \ s_{tx}^*(t)
\end{align}
$$
where
- $s_{\mathrm{tx}}(t) = \exp \left(j \ 2\pi \left(f_c t + \tfrac{1}{2} S t^2\right)\right)$ is the `transmitted signal`
- $s_{\mathrm{rx}}(t) = s_{\mathrm{tx}}(t - \tau)$ is the `received signal`

Simplifying it, we can get
$$
s_{\text{IF}}(t)
\approx \exp(j \ 2\pi (S \tau) \ t)
$$

> Note that $\text{IF signal}$ is a `pure tone` with frequency $f_{b}=S\tau$.
> Also $f_{b}$ is called the `beat frequency`.

---
`Computing the Delay`
Substituting in $\tau = \frac{2R}{c}$,
$$
\begin{align}
&f_{b}  
= S . \frac{2R}{c} \\[6pt]
\implies &R = \frac{c}{2S} . f_{b}
\end{align}
$$
This means that $\text{larger delay} \implies \text{larger beat frequency}$.

> `Fast-Fourier Transform` over $\text{fast time}$ gives range bins.
> This is why $\text{fast time} = \text{range dimension}$.

---
### 3. Estimating Velocity

`Range Model`
Suppose the target has radial velocity $v$.
Then,
$$
R_{k} = R_{0} + v\ kT_{c}
$$
where
- $R_{k}$ is the `target range` at $k^{th} \text{ chirp}$ 
- $R_{0}$ is the `initial range` at the first chirp
- $v$ is the `radial velocity` of the target. 
  $(\text{positive} \implies \text{moving away})$
- $k$ is the `chirp index`
- $T_{c}$ is the `duration` of one chirp

---
`Signal Across Chrips`
After fixing a range bin, we get
$$
s[k] = A \ e^{jk \ \Delta \phi}
$$
where
- $s[k]$ is the complex `IF signal sample` at chirp index $k$
- $A$ is the complex `amplitude`
- $j^2=-1$ is the `imaginary unit`

> Note that this is `discrete-time complex sinusoidal`.

Applying `FFT` across chirps $(\text{slow time})$, we can get `Doppler Frequency Bin` corresponding to $f_{D} = \frac{\Delta \phi}{2\pi \ T_{c}}$.

---
`Phase Difference`
Recall from [[Wave Phase]] that $\phi(R) = \frac{4\pi}{\lambda} R$ for reflected waves.

Substituting in the `range model`,
$$
\phi_{k} = \frac{4\pi}{\lambda}
(R_{0} + vk \ T_{c})
$$
Hence, the phase difference is defined as
$$
\Delta \phi
= \phi_{k+1} - \phi_{k}
= \frac{4\pi}{\lambda} \ vT_{c}
$$
---
`Doppler Frequency`
For an $\text{IF signal}$ $s_{IF}(k) \propto \exp(jk \ \Delta \phi)$, the corresponding Doppler frequency is
$$
f_{D} = \frac{\Delta \phi}{2\pi \ T_{c}}
$$

Substituting in $\Delta \phi$, we get
$$
f_{D} = \frac{2v}{\lambda}
\implies \boxed{ \ v = \frac{\lambda}{2} f_{D} \ }
$$
where
- $f_{D}$ is the `Doppler frequency` from `FFT`
- $v$ is the `radial velocity`
- $\lambda$ is the wavelength

---
### 4. Angle Estimation
With multiple antennas spaced by $d$, a target at angle $\theta$ causes a `path difference`:
$$
\Delta L = d\sin(\theta)
$$
where
- $\Delta L$ is the `path length difference`
- $d$ is the `antenna spacing`
- $\sin(\theta)$ is the projection of spacing onto wave direction $\theta$

This introduces a `phase difference` of
$$
\Delta \phi 
= \frac{2\pi d \ \sin(\theta)}{\lambda}
$$
By measuring phase across antennas using `FFT`, angle $\theta$ can be estimated.

---
## See Also
- [Youtube Video](https://youtu.be/xUGWHGjCtII?si=W6yVHdMw-MZkegUR)