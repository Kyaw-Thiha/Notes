# Matching Cost
#cv/object-detection/models/detr/matching-cost

This is the matching cost used to calculate whether the prediction class is same with the ground truth class, as well as how accurate the prediction boxes are to ground truth boxes.

This matching cost is used to fill up the cost matrix for [[Hungarian Algorithm]].

---
## Matching Cost Formula  
For prediction $\hat{y}_{\sigma(i)}$ matched to ground truth $y_i$:  
$$
L_\text{match}(y_i, \hat{y}_{\sigma(i)}) =
- \mathbf{1}_{\{c_i \neq \varnothing\}} \, \hat{p}_{\sigma(i)}(c_i)
+ \mathbf{1}_{\{c_i \neq \varnothing\}} \, L_\text{box}(b_i, \hat{b}_{\sigma(i)})
$$  
where
- $\hat{p}_{\sigma(i)}(c_i)$ is the predicted probability of the class.
- **Class cost**: if object exists ($c_i \neq \varnothing$), reduce the loss by the confidence of prediction.  
- **Box cost**: if object exists, compare $b_i$ and $\hat{b}$ (with $L_1 +$ GIoU).  
- If $c_i = \varnothing$ (no object), cost = constant (not dependent on prediction).  

---
## Box Loss 
$L_{1}$ Loss have different scales for small & large boxes, even if their relative errors are same.

To mitigate this, a linear combination of $L_{1}$ loss & generalized IoU is used for the box loss.


$$
L_\text{box}(b_i, \hat{b}) = \lambda_{iou} \, L_{iou}(b_i, \hat{b})
+ \lambda_{L1} \, \| b_i - \hat{b} \|_1
$$  
where
- $L_{iou}$ = generalized IoU (scale-invariant).  
- $L_1$ = absolute distance between box coordinates.  
- $\lambda_{iou}, \lambda_{L1}$ = hyperparameters to balance terms.  

## See Also
- [[DETR]]
- [[Hungarian Algorithm]]
- [[Hungarian Loss]]
