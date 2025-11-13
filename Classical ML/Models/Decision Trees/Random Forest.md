# Random Forest
#ml/models/classic/decision-tree/random-forest
This is a [[Decision Tree]] that uses [[Bagging]] and additional feature selection stochasticity to create `Decision Boundaries`, without overfitting.

![Random Forest](https://media.geeksforgeeks.org/wp-content/uploads/20250627112439534287/Random-forest-algorithm.webp)

---
## Algorithm

$$
\text{Random Forest} = \text{Bagging Decision Trees} + \text{Stochasticity}
$$

1. Carry out [[Bagging#`Bootstrap`|bootstrap]] sampling on the dataset.
2. Choose a subset of features to add `stochasticity`.
   - Each node selects $\sqrt{ n }$ features.
   - Hence, each tree gives you different predictions.
3. Carry out [[Bagging#`Aggregating`|aggregation]] (`majority vote`) to blur out the noise of each predictions.

---
## Why bootstrap?

This allows us to treat each individual [[Decision Tree]] as biased, high-variance estimator producing a sample mean.
By `Law of Large Number`, increasing the number of `DTs` reduces the `sample mean variance`.

## Why subset of feature?

If you have a very informative feature that dominates every split, all the `DTs` will have the same root node.
Hence, this lack of stochasticity will not help reduce the variance.
Choosing a subset of features also help with reducing correlation between `DTs` (which is important in `Information Theory`)

## Robustness
[[Bagging#`Bootstrap`|Bootstrap sampling]] allows `Random Forest` to explore small changes in data.
This essentially means that small changes in `training set` will have small impact on the output.

---
## See Also
- [[Decision Tree]]
- [[Bagging]]
