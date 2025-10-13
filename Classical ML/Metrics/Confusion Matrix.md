# Confusion Matrix
#ml/metrics/confusion-matrix

![Confusion Matrix](https://i0.wp.com/glassboxmedicine.com/wp-content/uploads/2019/02/confusion-matrix.png?fit=1200%2C675&ssl=1)

## Accuracy
$$
\begin{align}
\text{Accuracy} &= \frac{\text{Total No. of Correct Predictions}}{\text{Total No. of Predictions}} 
 \\[6pt]
&= \frac{TP + TN}{TP + TN + FP + FN}
\end{align}
$$

## Error
$$
\begin{align}
\text{Error} &= 1 - \text{Accuracy} 
 \\[6pt]
&= \frac{FP + FN}{TP + TN + FP + FN}
\end{align}
$$

## Precision
$$
\frac{TP}{TP + FP}
$$

## Recall
$$
\frac{TP}{TP + FN}
$$

## F1 Score
Used when you want a balance between precision and recall.
$$
F_{1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$
