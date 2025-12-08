# Hierarchical K-Means
#ml/classic-models/k-means/hierarchical 

`Hierarchical K-Means` build cluster centers in a hierarchical manner.

![Hierarchical K-Means](https://miro.medium.com/v2/1*oF00T9hCGpr6N-1zm2M1nw.gif)

---
`Motivation`
When querying a new data point's cluster,
- [[K-Means|Vanilla K-Means]] requires searching $K$ clusters
- `Hierarchical K-Means` require searching $\log K$ levels.

---
`Algorithm`

Start with the whole dataset.
Then, break it to $k_{i}$ $(k_{i}=2)$ clusters for each level $i$
Continue this till we get $K$ clusters.

This means that we will end up with $K$ clusters and $\log K$ levels.

---
## See Also
- [[K-Means]]
- [[Math behind K-Means]]