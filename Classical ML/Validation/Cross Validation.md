# Cross Validation

The models can have different hyper-parameters which can have effects on the accuracy of the model.

To test out these hyper-parameters, 
1. Split the dataset into - `train`, `valid`, and `test`.
2. Fit the model based on chosen hyper-parameter on the `train` dataset, and test its results on `valid` dataset.
3. Repeat the step-2 for all hyper-parameter
4. Choose the hyper-parameter with lowest loss
5. Now, fit the model on the chosen hyper-parameter on `train` + `valid` dataset.

## Problems
- Can be very time-consuming
- Need sufficient dataset size. Else, the underlying phenomena in `train` set may not exist in `valid` set or vice-versa.
- Need to use random partition (think about data collected over time)
- Over and under-fitting may still occur, for example if `valid` set is very small.
