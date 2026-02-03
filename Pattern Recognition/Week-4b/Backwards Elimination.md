# Backwards Elimination
The [[Forward Greedy Action Selection|Forward Method]] of [[Feature Selection]] progressively adds in features.
Instead, we can progressively eliminate features.

One way is to use `bootstrap methods` to train multiple [[Linear Predictor]].
Then, determine if coefficients are statistically indistinguishable from $0$ at certain confidence level.

We can then remove the feature, and repeat.

---
