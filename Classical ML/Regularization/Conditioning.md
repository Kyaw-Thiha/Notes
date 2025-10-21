# Conditioning

A `Condition Number` is the ratio of largest singular value and smallest singular value.

$$
cond(A) = \frac{\sigma_{max}(A)}{\sigma_{min} (A)}
$$

Large condition number means matrix significantly magnifies errors of the vectors.
In `ML`, large condition number often implies signs of overfitting.
