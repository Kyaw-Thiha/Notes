# Explained Variance Ratio

Based on [[Math Behind PCA]], we know that largest `eigenvalues` correspond to highest `reconstruction variance`.

![Explained Variance Ratio|400](https://www.researchgate.net/profile/Xavier-Olive/publication/329323302/figure/fig2/AS:736658181865482@1552644405989/Explained-variance-ratios-for-each-component-of-a-FPCA-decomposition-over-the-1398-track.ppm)
The lower the value of $\lambda_{j}$, the smaller its `reconstruction variance`.


---
`Cumulative Explained Variance Ratio`

$$
\rho 
= \frac{\sum^k_{j=1} \lambda_{j}}{\sum^N_{j=1} \lambda_{j}}
= \frac{\text{Variance in Principal Component Subspace}}{\text{Total Variance in Data}}
$$

`Cumulative Explained variance ratio` $\rho$ help us choose $top-k$ `eigenvalues`.

![Cumulative Explained Variance Ratio|400](https://saturncloud.io/images/blog/variance_ratio_plot.webp)
