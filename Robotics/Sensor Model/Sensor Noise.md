# Sensor Noise
#robotics/sensor/noise
`Sensor Noise` is the distortions and noise introduced by the [[Sensing Process]].

![Sensor Noise|400](https://miro.medium.com/v2/resize:fit:700/1*TCTUoHmNHekdE9gut89dqQ.png)

$$
s(k) = r(k) + n(k)
$$
where
- $s(k)$ is `final sampled signal` value
- $r(k)$ is `correct signal` value (undistorted, noise-free) 
- $n(k)$ is the `noise term`

---
`Reasonable Assumptions`
- Noise values are `uncorrelated`.
  Noise component for $s(j)$ does not depends on $s(k)$
- Noise is `zero-mean`.
  Average of noise values should approach $0$.

> Do note that if you have information about sensor noise (for example through `sensor calibration`), then you should use it.

---
## Noise Removal
Using the reasonable assumptions above, we have $2$ main methods for noise removal:

1. `Averaging Multiple Readings`
2. `Local Smoothing (Linear Filtering / Convolution)`

---
### Averaging Multiple Readings

![Averaging Multiple Readings|500](https://www.researchgate.net/profile/Lukas-Koester-3/publication/346562138/figure/fig2/AS:964208837537793@1606896709299/Noise-reduction-due-to-averaging-over-multiple-images-as-noted-in-the-picture-An-image.png)

This is best used when 
- we have `multiple redundant sensors` reading the same quantity 
- or we have a sensor which have `relatively fast read-rate` with respect to the speed of change of signal.

`Proof`
The sensor can be modelled as $s(k) = r(k) + n(k)$.
Averaging the readings, we get
$$
\begin{align}
\frac{1}{N} \sum^N_{k=1} s(k)

&= \frac{1}{N} \sum^N_{k=1} \left( r(k) + n(k) \right) \\[6pt]

&= \frac{1}{N} \sum^N_{k=1} r(k) + \frac{1}{N} \sum^N_{k=1} n(k)  \\[6pt]

&= \frac{1}{N} \sum^N_{k=1} r(k) + \epsilon  
& \text{by zero-mean assumption} \\[6pt]

&= \frac{1}{N} N \times r(k) + \epsilon \\[6pt]

&\approx r(k)
\end{align}
$$

---
### Local Smoothing
For fast moving signals, denoising with averaging over samples is not possible.
So, `temporal/spatial proximity` is used to average out the noise.

> **Note**: This process unavoidably destroy some information present in the original signal.

`Using Filters`

![Filters|300](https://miro.medium.com/1*tvgDlagu7Tm7q7Y2Y6Gs0g.png)
We can use filters like `Gaussian filter` to remove noise from the signal.

![Denoising using filters|400](https://miro.medium.com/v2/resize:fit:700/1*0Wnbc_ydlhNZs3weYHuhfg.png)

Note that `Sample Averaging` is better than using filters.
So when we are unable to provide multiple readings of a slow moving signal, we can consider `Weighted Averaging`.

---

`Linear Filter & Convolution`

Weighted averaging on a signal can be done by applying a `linear filter` to the signal.
The mathematical operation performed is called a `convolution`.
The array with values used by filter is called a `kernal`.

[Image here]

To obtain the filtered output at position $j$,
1. Place the filter kernal centered at $j$
2. Multiply the corresponding overlapping entries from signal and filter
3. Add up the resulting values

Mathematically, we can define a convolution operation as
$$
O(j) = \sum^{N/2}_{k=-\frac{N}{2}} I(j+k) 
\ \cdot \ h(-k)
$$
where
- $O(j)$ is the output at position $j$
- $I(j+k)$ is the input at position $j+k$
- $h(-k)$ is the filter kernal

Notice that the `filter kernals` are applied in reverse order$(-k)$.
This is equivalent to multiplying overlapping entries between a signal and a flipped kernal.

This is required to preserve an important property of convolution: 
> If we applied filter $h(x)$ where $x=\delta(j)$ is a `unit impulse function`, the output should be exactly the same filter kernal $h(x)$.

Note that
$$
\delta(j) = \begin{cases}
1 & \text{if } j=0 \\[6pt]
0 & \text{otherwise}
\end{cases}
$$

---
`Correlation`
If the filter is not flipped, the `convolution` would result in a flipped output.
This operation is called `correlation`.

In practice, the filters used are usually `horizontally symmetric`, in which case `convolution` and `correlation` produces the same output.

---
## Read More
- [[Sensor Model]]
- [[Sensor]]
- [[Sensing Process]]