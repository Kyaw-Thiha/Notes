`Event-based Camera` $(\text{a.k.a} \text{ Neuromorphic Camera})$ are bio-inspired sensors that 
- independently record pixel-level brightness changes$(\text{events})$ 
- instead of relying on a frame-rate.

![Event Camera|500](https://moonlight-paper-snapshot.s3.ap-northeast-2.amazonaws.com/arxiv/recent-event-camera-innovations-a-survey-4.png)

---
## Event Representation
An `event` is denoted as a tuple $(\mathbf{u}, t, p)$.
This is triggered when `logarithmic intensity` $L$ surpasses the threshold $C$.

This can be denoted as
$$
p = \begin{cases}
+1 & , \ L(\mathbf{u}, t) - L(\mathbf{u}, t - \Delta t) \geq C \\[6pt]
-1 & , \ L(\mathbf{u}, t) - L(\mathbf{u}, t- \Delta t) \leq -C \\[6pt]
0 &, \ \text{other}

\end{cases}
$$
where 
- $\mathbf{u} = (x, y)$ is the `pixel location`
- $t$ is the `timestamp`
- $p\in \{ -1,1 \}$ is the `polarity` $(\text{sign of brightness})$
- $\Delta t$ is the `time interval` since last event at pixel $\mathbf{u} = (x,y)$

**Stream of Events**
A stream of events can then be denoted as
$$
\mathcal{E}
= \{ e_{i} \}^N_{i=1}
= \{ \mathbf{u}_{i}, \ t_{i}, \ p_{i} \}
, \ i \in N
$$

#### Representation Method
These events can be represented in different methods for processing inside [[Neural Network|neural networks]].
- Image-Based
- Surface-Based
- Voxel-Based
- Graph-Based
- SNN-Based
- Learning Based

[[Event Camera Representation|Read More]]

---
