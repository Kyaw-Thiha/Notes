# Intersection over Union (IoU)
#cv/object-detection/iou 

`IoU` measures how much the two bounding boxes overlap relative to their total area.

![IoU](https://learnopencv.com/wp-content/uploads/2022/06/Understanding-Intersection-Over-Union-in-Object-Detection-and-Segmentation.jpg)

Let $y$ be the actual area, and $\hat{y}$ be the predicted area.
Then,
$$
\text{IoU} 
= \frac{\hat{y} \ \cap \ y}{\hat{y} \ \cup \ y}
$$

---
