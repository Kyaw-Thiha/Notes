# Contamination Delay
[[Contamination Delay|Contamination delay]] is the minimum time from when an input changes until any output starts to change its value.

![Contamination Delay|250](https://computationstructures.org/lectures/cmos/slides/Slide16.png)

---
## Calculating Contamination Delay
- Find the `short path` (path with smallest number of gates).
- Sum up the contamination delay of all gates on the short path.

---
[[Contamination Delay|Contamination delay]] typically should be lower-bounded.
We can make it longer by adding [[Tri-State Buffer|buffers]] to the `short path.`