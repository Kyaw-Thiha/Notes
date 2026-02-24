# ByteTrack
#cv/object-tracking/algorithm/bytetrack

`ByteTrack` considers association of low-score detections in order to solve false negatives during occlusion.

![ByteTrack Association|300](https://cdn.prod.website-files.com/67d7ebcca3cabb623904dcde/67d7f31782fc546599bcf291_64588bc23b78b730227108d3_f9dc28fb.png)

In [[SORT]], low scoring detections are eliminated by threshold.
`ByteTrack` uses similarity between tracklets and low-score detection boxes to distinguish between objects and background.

---
## Byte Algorithm

1. Separate detections and high-score and low-score.
2. First, associate high score detections to the tracklets.
3. Then, associate low score detections to unmatched tracklets.
   This recover the objects in low score detection boxes and filter out background simultaneously.

---
## See Also
- [Original Paper](https://arxiv.org/abs/2110.06864)
- [[Object Tracking]]