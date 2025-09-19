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

## Cluster Assignment Matrix (Binary)
The cluster assignment matrix can be represented as $L \in \{0, 1\}^{N \times K}$ 
$$
l_{ij} =
\begin{cases}
1, & \text{if } y_i \text{ is assigned to cluster } j \\
0, & \text{otherwise}
\end{cases}
$$

such that 
- Row sum $\sum^K_{j=1} l_{i, j} = 1$
- Column sum $\sum^K_{i=1} l_{i, j} = \text{No. of points assigned to cluster-}j$
- Sum of all elements $\sum^N_{i=1}\sum^K_{j=1} l_{i, j} = \sum^K_{j=1}\sum^N_{i=1} l_{i, j} = N$

## Objective Function
The objective function measures how close the points are grouped to their clusters.

For each cluster, we sum the `mean-squared distance` between the points to the center.
Then, we sum up all these distances from each clusters.
$$
E(L, \{c_{j}\}^K_{j=1}) = \sum_{i=1}^{n} \sum_{j=1}^{k} l_{ij} \, \| x_i - c_j \|^2
$$

$L = argmin_{L} \ E(L, \{c_{j}\}^K_{j=1})$ 
To try to minimize the objective function, for each point, we measure its distance to all the centers, and assign it to the closest cente.

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

## K-Means Convergence Problem
The objective function of `K-Means` has local minimum.
So, we need to ensure a good initialization of center points.

## Choosing the value of K
Compute the `Withing-Cluster Sum of Squares (WCSS)`, which is basically the objective function, over different values of k.

Note that we calculate this after `convergence`.

![Elbow Method](https://miro.medium.com/v2/resize:fit:1340/1*BKKH21zsY1vAomA7FxQGHg.png)


## Proof of Objective Function
`WTS: ` $c_{j} = argmin_{c_{j}}E(L, c_{j})$
`Objective Function`
$$
\begin{align}
E(L, c_{j})  
&= \sum^N_{i=1} l_{i,j}.||y_{i} - c_{i}||^2  \\
&= \sum^N_{i=1} l_{i,j}.(y_{i} - c_{j})^T . (y_{i} - c_{i}) \\
&= \sum^N_{i=1} l_{i,j}.(y_{i}^T.y_{i} - 2.y_{i}^T.c_{j} + c_{j}^T.c_{j}) \\
\end{align}
$$
`Minimizing Objective Function`
$$
\begin{align}
\frac{\partial E}{\partial c_{j}} &= 0 \\[4pt] 

\sum^N_{i=1}.l_{i, j}.(-2y_{i} + 2c_{j}) &= 0 \\[4pt] 

\sum^N_{i=1}.l_{i, j}.y_{i} &= \sum^N_{i=1}.l_{i, j}.c_{j}  \\
\\[4pt]

\sum^N_{i=1}.l_{i, j}.y_{i} &= .c_{j}.\sum^N_{i=1}.l_{i, j}  \\
\\[4pt]

c_{j} &= \frac{\sum^N_{i=1}.l_{i, j}.y_{i}}{\sum^N_{i=1}.l_{i, j}}
\end{align}
$$

