# Hyperparameter
[[Hyperparameter|Hyperparameters]] are [[Neural Network|neural network]] settings that are not optimized during [[Neural Network|learning]].

Examples include:
- Batch Size
- No. of Layers
- Layer Size
- [[Activation Function|Type of Activation Function]]
- [[Learning Rate]]

---
## Hyperparameter Optimization
We use the validation dataset to tune the hyperparameters.
![image|400](https://notes-media.kthiha.com/Hyperparameters/ec29620fa68568a8162ecfbbc0b4c8cb.png)

---
## Tuning Techniques
There are different techniques for tuning [[Hyperparameter|hyperparameters]] namely
- Grid Search
- Random Search
- Bayesian Optimization

![300](https://miro.medium.com/v2/1*xaxUIGnNZ6P5ExaFmhkTog.jpeg)

Most hyperparameter tuning is done with libraries like optuna which uses a form of Bayesian optimization.

---