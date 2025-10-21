# K-Fold Cross Validation

![K-Fold Cross Validation](https://miro.medium.com/v2/resize:fit:1400/1*GhKMAUmi4bfFiEwZCPlDsA.png)
This is a validation method that can be used when the dataset size is small.

1. Partition data into $k$ subset
2. For each subset, the model train on the remaining (k-1) subsets, before being tested on the chosen subset.
3. The hyper-parameters with the lowest loss is chosen.
4. These hyper-parameters are then used to train on train + test datasets.

## No. of Models
Let 
- $m$ be the number of hyper-parameters
- $C$ be the no. of values to test per hyper-parameters
- $k$ be the no. of `valid` dataset

Then,
- For cross validation: $C^m$ models
- For k-fold cross-validation: $k.C^m$ models
