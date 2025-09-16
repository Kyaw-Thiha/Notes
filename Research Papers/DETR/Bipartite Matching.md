# Bipartite Matching
#cv/object-detection/bipartite-matching 

![Bipartite Matching](https://www.researchgate.net/publication/370653583/figure/fig5/AS:11431281365292443@1744213021454/DETR-bipartite-matching-loss-to-find-the-best-match-between-the-ground-truth-and-the.tif)
Models like [[DETR]] has a fixed number of output predictions (eg: 100 bounding boxes).
But actual image always have a lower number of ground truth objects.

This is where bipartite matching come in - it matches predicted boxes to their corresponding ground truth objects.

## Bipartite Graph
![Bipartite Graph](https://i.ytimg.com/vi/GhjwOiJ4SqU/maxresdefault.jpg)
Bipartite graph is a graph whose nodes can be split into two partitions such that
- Every edge goes from one group to another
- No edges connect nodes within the same group.

## Bipartite Graph Visualization
![Birpartite Graph](https://miro.medium.com/v2/0*PWKRBToddj9I36q4.gif)

## DETR Case
- Partition-1: Predictions
- Partition-2: Ground truth objects

Note that
- Every ground truth is matched to exactly one prediction
- Each prediction is used at most once
- Unmatched predictions get matched to special "no object" label

## See Also
- [[DETR]]
- [Bipartite Graph](https://python.plainenglish.io/bipartite-graphs-a-fundamental-concept-in-graph-theory-779ddf45218d)
