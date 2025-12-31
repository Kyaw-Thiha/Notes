# Global Positioning System
#robotics/localization/gps #robotics/localization/gns
`GPS`/`GNS` is a satellite based network providing precise location, navigation and timing data (`PNT data`) using trilateration.

![GPS|500](https://miro.medium.com/v2/resize:fit:1400/1*fbPXH2Yk74l72WsRBvRGaQ.gif)

---
## How GPS Works

`Forming the sphere`
1. Each GPS Satellite emits a signal with information about the satellite's `position and timestamp`. 
2. The GPS receiver calculates the `time-of-travel` from the GPS satellite using the timestamp data.
3. Time of travel is used to compute the `distance` to the satellite. 
   The receiver must be somewhere on a `sphere` centered at the satellite, whose radius is the computed distance.

---
`Trilateration`
Trilateration is core mechanism in GPS to determine the location.

As outlined above, the `distance sphere` for each of the satellite is computed from `time-of-travel`.

| ![1 Satellite](https://oceanservice.noaa.gov/education/tutorial_geodesy/media/geo09b1_700.jpg) | ![2 Satellites](https://oceanservice.noaa.gov/education/tutorial_geodesy/media/geo09b2_700.jpg) |
| ---------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |

Intersection of 3 satellite distance spheres allows the `position` $(\text{latitude, longitude, altitude})$ to be determined.

The fourth satellite can be used to `correct clock errors` for greater accuracy.

| ![3 Satellites](https://oceanservice.noaa.gov/education/tutorial_geodesy/media/geo09b3_700.jpg) | ![4 Satellites](https://oceanservice.noaa.gov/education/tutorial_geodesy/media/geo09b4_700.jpg) |
| ----------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |

This process can be visualized as
![Trilateration](https://upload.wikimedia.org/wikipedia/commons/2/27/GPS24goldenSML.gif)


---
`Limitations`
- Earth is not a perfect sphere, so it does not have a uniformly distributed mass.
  This causes gravitational variations, leading to imperfect ellipsoid satellite orbit.
  The satellite clock is affected by these `height variations`.
- `Atmospheric conditions` affect the signal's time of travel.
- No `direct-line-of sight` $(\text{Think of urban area})$.
- Difficult `synchronization` between satellite and receiver clocks
- Does not work `indoors`

---
## See Also
- [Detailed Article](https://www.telesens.co/2017/07/17/calculating-position-from-raw-gps-data/)
- [[Localization]]
