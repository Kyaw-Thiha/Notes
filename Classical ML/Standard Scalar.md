# Standard Scalar

When fitting a multi-dimensional $x$ vector to a model, the larger values of data in one dimension could dominate the result. 

So, we scale the data to normalize it.

$$
z = \frac{x - \mu}{\sigma}
$$
where
- $\mu$ is the mean
- $\sigma$ is the variance
Hence, the transformed data has mean = 0 and standard deviation = 1.

### Before
![[no-scaling.png]]

### After
![[after-scaling.png]]

This helps
- Prevent one feature from dominating the result
- Gradiant-descent based algorithm converge faster when data is scaled
- Distance in clustering and PCA becomes more meaningful.

## Important
You must only fit the StandardScaler onto the training set.
Then, you use that fitted scaler to apply transform onto the training, validation & test datasets.

During inference, you must use the same scaler to apply transforms.

```python
scaler = StandardScaler()
scaler.fit(X_train)        # compute μ and σ only on train

X_train_scaled = scaler.transform(X_train)
X_val_scaled   = scaler.transform(X_val)   # use train’s μ, σ
X_test_scaled  = scaler.transform(X_test)  # use train’s μ, σ
```
