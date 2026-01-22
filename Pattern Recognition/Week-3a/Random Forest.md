## Random Forest
#ml/ensemble/random-forest
1. Create many subsamples of the dataset, $S' \subset S$ by sampling uniformly with replacements.
2. For each dataset, choose $k$ features features from the dataset uniformly.
3. Train a decision tree on each dataset $(S', \{ i_{1}, \dots, i_{k} \})$ 
4. To label them, take the majority vote of each of the trees in the forest.

---
`Decision Stumps`

A `decision stump` is a [[Decision Tree]] on $k=1$ features.
It only has one root node, with two leaf nodes.

---

## Ensemble Methods
Combine [[Weak Learner|weak learners]] into one classifier that is better than any of them.

[[AdaBoost]] makes a strong classifier by applying a linear predictor to the outputs of a collection of weak learners.
The composition of linear classifiers increases their expressivity.

---
