## Surface-Based Representation
`Surface-based Active Events (SAE)` maps event streams to a `time-dependent` `surface tracking` `temporal-spatial activity` around each event's location.

> Instead of intensity-based context (image methods), it uses **temporal surface geometry** to encode information.

**Mathematical formulation**
For each $i^{th}$ event $e_{i}$,
$$
\tau_{i} ([x_{n}, y_{n}]^T, \ p)
= \max_{j\leq i}
\{ t_{j} \mid [x_{i} + x_{n}, \ y_{i} + y_{n}], 
\ p_{j} = p \}
$$
where
- $x_{n} \in \{ -r, r \}$ is the `horizontal offset` from event $e_{i}$
- $y_{n} \in \{ -r, r \}$ is the `vertical offset` from event $e_{i}$
- $p_{j} \in \{ -1, \ 1 \}$ is the `polarity` of event $e_{j}$
- $t_{j}$ is the `timestamp` of event $e_{j}$
- $r$ is the `neighbourhood radius`

> For each location in a $(2r +1) \times (2r+1)$ neighbourhood around $e_{i}$,
> it stores the timestamp of the most recent event with matching polarity.

Note that it has a normalization problem.
Timestamps monotonically increase $[0, \infty]$, but unbounded values are unsuitable for [[Neural Network]].

---
