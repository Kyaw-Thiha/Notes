# Interview Notes

`Precision vs Recall`
![Precision vs Recall](https://towardsdatascience.com/wp-content/uploads/2021/11/1xMl_wkMt42Hy8i84zs2WGg.png)

`Confusion Matrix`
![Confusion Matrix](https://www.mygreatlearning.com/blog/wp-content/uploads/2020/03/confusion0-matrix-viosualtion-1024x661.webp)

`IoU`/`NMS` 

[[Non-Maximum Suppression (NMS)]]
![NMS example](https://miro.medium.com/v2/1*4IOA6RbJFt59IPglC1LjRg.png)

`mAP`
[[Mean Average Precision (mAP)|Read Here]]

`Metrics`
- [[HOTA]]
- [[IDF1]]
- [[MOTA]]
- [[MOTP]]
- [[Mean Average Precision (mAP)]]

`Obj Tracking`
- `SORT`
- `DeepSORT`: uses appearance CNN for occlusion
- `ByteTrack`: helps with low-confidence pred
- `OC-SORT`: observation centric to help with non-linear movement
- `BoT-SORT`: assumes camera movement (improved Kalman)
- `SMILETrack`: stabilize association, handling detector noise better

`Datasets`
- DSEC
- MVSEC

`Drone Metrics`
Around 0.8 on mAP@0.5
Around 0.5 on mAP@O.5-0.95
Post detection filtering

`Questions`
- Which direction would you see neuromorphic  
- Why robotics to neuromorphic
- I noticed HDC has a lot of appealing properties—noise robustness, fast operations, low-power computation—but it isn’t used as widely as deep learning for things like object detection or large-scale models. From your perspective, what factors have limited its adoption so far?
- What kind of infrastructure does the lab have?


https://furlong.gitlab.io/
