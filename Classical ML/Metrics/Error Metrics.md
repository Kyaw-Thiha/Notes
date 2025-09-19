# Common Error Metrics

- **Mean Squared Error (MSE)**
  $$
  \text{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2
  $$

- **Root Mean Squared Error (RMSE)**
  $$
  \text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}
  $$

- **Mean Absolute Error (MAE)**
  $$
  \text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|
  $$

### Metrics Comparison

| Property                  | MAE | MSE | RMSE |
|----------------------------|:---:|:---:|:----:|
| Robust to Outliers         | ✅  | ❌  | ❌   |
| Sensitive to Large Errors  | ❌  | ✅  | ✅   |
| Easy to Interpret          | ✅  | ❌  | ✅   |
| Mathematically Convenient  | ❌  | ✅  | ✅   |
| Same units as target       | ❌  | ❌  | ✅   |
| Penalizes squared errors   | ❌  | ✅  | ✅   |

