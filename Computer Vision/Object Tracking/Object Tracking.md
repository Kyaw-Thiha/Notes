# Object Tracking
#cv/object-tracking

`Object Tracking` is a computer vision technique that track different object across a set of video frames.

![Object Tracking](https://learnopencv.com/wp-content/uploads/2022/06/Multiple-Object-Tracking-using-DeepSORT.gif)
Essentially, we are carrying out `object detection`, assigning trackID, and updating them over time.

---
## Main Paradigms

`Tracking-by-Detection (TbD)`
First, carry out `object detection`, then track them (association & filtering).

`End-to-End Tracking`
A single end-to-end model handles both detection & tracking jointly.
This is the latest technique used by `Transformer-based` models.


---
## Main Concepts


![Object Tracking|500](https://www.researchgate.net/publication/358134782/figure/fig1/AS:11431281350550979@1743777217551/Overview-of-the-Kalman-Filter-based-object-tracking-algorithms-used-in-this-work.tif)


`Data Association`
Connecting detections in the current frame to existing tracked objects.

`Track Lifecycle Management`
Maintaining stable track IDs despite noise, occlusions, and missed detections.

`Motion Predictions`
Using `Kalman Filter` to estimate where each tracked object will likely move by the next frame.

---
## Key Models
- `SORT`
- `DeepSort`
- `BYTETrack`
- `OC-SORT`/`Strong-SORT`
- `Transformer Trackers`

---
## Evaluation Metrics
- [[HOTA]]
- [[IDF1]]
- [[MOTP]]
- [[MOTA]]

## See Also
- [Very Good Literature Review](https://arxiv.org/abs/2506.13457)