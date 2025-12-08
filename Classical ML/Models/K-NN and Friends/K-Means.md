# K-Means
#ml/classic-models/k-means
This is a clustering method that make clusters based on k-nearest neighbours.

![K-Means|400](https://media.geeksforgeeks.org/wp-content/uploads/20190812011831/Screenshot-2019-08-12-at-1.09.42-AM.png)

## Steps

1. Choose the $k$ parameter
2. Select random $k$ number of centroids
3. Each data point is assigned to clusters whose centroid is closest to it (based on Euclidean distance)
4. For each cluster, the mean of the assigned data point is calculated.
5. For each cluster, the centroid is updated to the new mean.
6. Repeat the steps-3 to 5 till the centroids stop moving (convergence).

---
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

where
- $x_{i}$ is the data point at $i$
- $c_{j}$ is the cluster point at $j$
- $l_{ij}$ is the cluster assignment `indicator function`

$L = argmin_{L} \ E(L, \{c_{j}\}^K_{j=1})$ 
To try to minimize the objective function, for each point, we measure its distance to all the centers, and assign it to the closest cente.

[[Math behind K-Means|Read More]]

---
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
![K-Means++|500](https://miro.medium.com/v2/1*JoSI7oFZXqKG8HNti0asvg.gif)

1. Chooose 1 data-point at random
2. Compute distances
3. Pick next center with probability proportional to distance squared
   $P(y_i) = \frac{D(y_i)^2}{\sum_{k} D(y_k)^2}$
4. Repeat steps-2 and 3 till $k$ centers are chosen
5. Carry out normal K-means

---
## K-Means Convergence Problem
In `K-Means`, we alternate between 2 different optimization problems - one for updating the centroids, and another for assigning points to them.
Since each optimization problem reduces the objective function, and objective function's lower bound is $0$, the algorihm is guaranteed to converge.

However, the objective function of `K-Means` has local minimums.
So, we need to ensure a good initialization of center points.

---
## Choosing the value of K

`Elbow Method`
Compute the `Within-Cluster Sum of Squares (WCSS)`, which is basically the objective function, over different values of k.

Note that we calculate this after `convergence`.

![Elbow Method|500](https://miro.medium.com/v2/resize:fit:1340/1*BKKH21zsY1vAomA7FxQGHg.png)

`Penalized Likelihood`
This is form of a `Bayesian Model Selection` 

We allow $K$ to grow as large as wanted.
Then, we add penalty term to the [[#Objective Function]] in order to penalizes the no. of clusters.

`Latent Dirichlet Analysis (LDA)`
Using a `generative probabilistic model` to discover hidden topics(cluster centers) inside a document.

![LDA|500](https://cdn.analyticsvidhya.com/wp-content/uploads/2021/06/Graphical-model-of-latent-Dirichlet-allocation-LDA.webp)


---
## K-Means Variants

`K-Metoids`
Instead of using the `mean` to represent the cluster, use the point closest to mean to represent the cluster.

`Hierarchical K-Means`
Build cluster centers in hierarchical fashion for efficient querying.
[[Hierarchical K-Means|Read More]]

`Mixture Model`
Represent the cluster by `probability distribution` of data.
Allow each point to represent `multiple probability distribution`.
[[Mixture Model|Read More]]

---
## See Also

- [[Math behind K-Means]]
- [[Curse of Dimensionality]]
- [[Voronoi Diagram]]