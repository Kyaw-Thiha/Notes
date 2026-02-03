# Linear Predictor

There are different kinds of `linear predictors` such as
- `Classifiers`: $h: \mathcal{X} \to \{ -1, 1 \}$
- `Regression`: $h: \mathcal{X} \to \mathbb{R}$
- `Classifiers`: $h: \mathcal{X} \to (\ \mathcal{Y} \to [0, 1] \ )$

We can express the space of linear predictors as
$$
L_{d} 
= \{ \ x \mapsto \langle w,x \rangle + b: \quad 
w\in \mathbb{R}^d, \ b \in \mathbb{R} \ \}
$$
