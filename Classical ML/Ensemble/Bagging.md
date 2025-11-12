# Bagging
#ml/ensemble/bagging 

This is an [[Ensemble Model]] designed to `reduce variance` and `prevent overfitting`.

![Bagging](https://media.licdn.com/dms/image/v2/D5612AQFNuVROwlbA0g/article-cover_image-shrink_600_2000/article-cover_image-shrink_600_2000/0/1715981791085?e=2147483647&v=beta&t=XdgIuX-pc_vy7vvlnxUT_AFifIfuvNbBd-soU4UF1Ms)

---
## Technique

### `Bootstrap`
Randomly draw $N$ samples with replacements from the dataset. 
This mean some data will be duplicated, while others are not chosen.
Repeat this sampling for $k$ times to train $k$ models.

### `Aggregating`
Train each of the $k$ model using the bootstrapped model (in parallel).
Then, do `majority voting`, or taking the `mean`.

## Example
- Decision Tree

---
## Limitations
- Does not reduce `bias`
- Require more computation
- Not ideal for small dataset, since `bootstrapping` reduces diversity

---
## See Also
- [[Ensemble Model]]

