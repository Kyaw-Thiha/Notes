# K-Means
This is a clustering method that make clusters based on k-nearest neighbours.

![K-Means](https://media.geeksforgeeks.org/wp-content/uploads/20190812011831/Screenshot-2019-08-12-at-1.09.42-AM.png)

## Steps

1. Choose the $k$ parameter
2. Select random $k$ number of centroids
3. Each data point is assigned to clusters whose centroid is closest to it (based on Euclidean distance)
4. For each cluster, the mean of the assigned data point is calculated.
5. For each cluster, the centroid is updated to the new mean.
6. Repeat the steps-3 to 5 till the centroids stop moving (convergence).

## Choosing Center
The choice of initial centroids have huge performance effect on the algoritms.
Can also end up in local minimum.
So, here are a few variants of centroids choosing
- **Random Labelling**: For first iteration, instead of assigning data point to nearest clusters, they are assigned to random cluster.
- **Random Initial Centers**: Choosing random positions as centers (not necessarily data points)
- **Random data point**
- **Multiple Restarts**: Essentially do multiple runs, and choose the run the lead to best centers.
- **K-Means++**

### K-Means++
1. Chooose 1 data-point at random
2. Compute distances
3. Pick next center with probability proportional to distance squared
   $P(y_i) = \frac{D(y_i)^2}{\sum_{k} D(y_k)^2}$
4. Repeat steps-2 and 3 till $k$ centers are chosen
5. Carry out normal K-means
