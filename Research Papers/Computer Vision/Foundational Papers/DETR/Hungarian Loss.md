# Hungarian Loss
#cv/object-detection/hungarian-loss 

After predictions cost matrix is computed through [[Matching Cost]], and prediction-ground truth pairs are generated through [[Hungarian Algorithm]], we use the `Hungarian Loss` to calculate the training loss.

## Hungarian Loss Formula

$$
L_{\text{Hungarian}}(y,\hat y)
=\sum_{i=1}^{N}\Big[
-\log \hat p_{\hat\sigma(i)}(c_i)
+\mathbf{1}_{\{c_i\neq \varnothing\}}\;L_{\text{box}}\!\left(b_i,\hat b_{\hat\sigma(i)}\right)
\Big]
$$  
- $N$: number of prediction slots.  
- $y$: ground truth set (class $c_i$, box $b_i$).  
- $\hat y$: model predictions (class probabilities $\hat p_j$, box $\hat b_j$).  
- $\hat\sigma$: assignment from Hungarian algorithm.  
- $-\log \hat p_{\hat\sigma(i)}(c_i)$: class loss for the matched pair (down-weighted if $c_i=\varnothing$).  
- $\mathbf{1}_{\{c_i\neq \varnothing\}}$: ensures box loss only applies to real objects.  

---
## Box Loss

$$
L_{\text{box}}(b,\hat b)
=\lambda_{iou}\,L_{iou}(b,\hat b)
+\lambda_{L1}\,\|b-\hat b\|_{1}
$$  

- $L_{iou}$ = generalized IoU (scale-invariant).  
- $L_1$ = absolute distance between box coordinates.  
- $\lambda_{iou}, \lambda_{L1}$ = hyperparameters to balance terms.  

Note that these terms are normalized over the number of true objects in the batch.  

---
## Class Balancing
Note that since `DETR` makes 100 predictions per image, most of the predictions are of $\varnothing$.

To prevent the model from collapsing into always predicting ∅, the classification loss for these cases is down-weighted:  

$$
L_{\text{class}}(i) =
\begin{cases}  
-\log \hat p_{\hat\sigma(i)}(c_i), & c_i \neq \varnothing \\[6pt]  
\frac{1}{10} \, \big(-\log \hat p_{\hat\sigma(i)}(\varnothing)\big), & c_i = \varnothing  
\end{cases}
$$  

- **$c_i \neq \varnothing$ (object)** → full weight (1.0).  
- **$c_i = \varnothing$ (no object)** → down-weighted by factor $0.1$.  

This way:  
- Real objects dominate the gradient signal.  
- The model does not trivially minimize loss by always predicting “no object.”  


---
## See Also
- [[DETR]]
- [[Matching Cost]]
- [[Hungarian Algorithm]]