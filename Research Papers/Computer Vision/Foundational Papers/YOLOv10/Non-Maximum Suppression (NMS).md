# Non-Maximum Suppression (NMS)
#cv/object-detection 
![NMS example](https://miro.medium.com/v2/1*4IOA6RbJFt59IPglC1LjRg.png)
## Definition  
**Non-Maximum Suppression (NMS)** is a post-processing algorithm used in object detection to eliminate redundant bounding boxes. 

When multiple bounding boxes overlap heavily and represent the same object, NMS keeps the one with the highest confidence score and suppresses the rest.

---

## Key Idea  
- Object detection models (e.g., YOLO, Faster R-CNN) often predict multiple bounding boxes for the same object.  
- **NMS selects the best box** based on confidence scores and suppresses others that overlap significantly.  
- Overlap is measured using **Intersection over Union (IoU)**.  

---

## Algorithm Steps  
1. **Input**: set of bounding boxes with confidence scores.  
2. **Sort**: order boxes by descending confidence score.  
3. **Select**: choose the box with the highest score.  
4. **Suppress**: remove boxes with IoU above a threshold relative to the selected box.  
5. **Repeat**: continue until no boxes remain.  

---

## Equation: Intersection over Union (IoU)  
$$
IoU = \frac{Area(B_{pred} \cap B_{gt})}{Area(B_{pred} \cup B_{gt})}
$$

- $B_{pred}$ = predicted bounding box  
- $B_{gt}$ = ground truth bounding box  

---

## Example  
Suppose an object detector outputs 5 bounding boxes around a cat with scores `[0.95, 0.90, 0.80, 0.60, 0.55]`.  
- After sorting, the highest-scoring box (0.95) is kept.  
- Any box with IoU > threshold (e.g., 0.5) relative to this box is suppressed.  
- Process repeats until all boxes are processed.  

Result: only **1–2 boxes remain**, avoiding duplicate detections.  

---

## Variants of NMS  
- **Hard NMS**: strict suppression, removes all overlapping boxes above threshold.  
- **Soft-NMS**: reduces confidence scores of overlapping boxes instead of discarding them.  
- **DIoU-NMS, CIoU-NMS**: use advanced IoU variants that consider distance and aspect ratio.  
- Latest models like **YOLOv10** and **DETR** models have adopted 'end-to-end' object detection which does not need NMS.

---

## Applications  
- Object detection pipelines (YOLO, SSD, Faster R-CNN).  
- Used in both training (optional) and inference (mandatory) stages.  
- Critical for producing clean, interpretable detection outputs.  

---

## References  
- [Ultralytics Glossary: Non-Maximum Suppression](https://www.ultralytics.com/glossary/non-maximum-suppression-nms)  
- Original NMS concept in computer vision: *Neubeck & Van Gool, 2006*.  

