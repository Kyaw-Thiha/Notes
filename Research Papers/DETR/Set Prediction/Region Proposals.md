# Region Proposals
#cv/object-detection/set-prediction/region-proposals 

A two-stage detection method where the model first proposes a large set of candidate regions, and then refines them into bounding boxes.

![Region Proposals](https://b2633864.smushcdn.com/2633864/wp-content/uploads/2020/06/region_proposal_object_detection_output_beagle_before.png?lossy=2&strip=1&webp=1)

## Algorithm
1. The model first generates ~1k-2k candidate boxes.
2. Then, a classifier decides what object (if any) are inside the specific box.

## Proposal Generation
- `R-CNN` (2014) and `Fast R-CNN` (2015) used handcrafted **Selective Search** which propose boxes based on colors, textures & edges.
- `Faster R-CNN` (2015) used a small CNN network called **Region Proposal Network** (RPN) which produce thousands of proposals per image.

## Related Models 

`R-CNN` (2014)
- Used selective search
- Each region is cropped and passed through CNN + SVM classifier.

`Fast R-CNN` (2015)
- Used selective search
- Improved speed by running CNN once on whole image, then cropping features with ROI pooling

`Faster R-CNN` (2015)
- Used small CNN network called Region Proposal Network (RPN)
- RPN is trainable so, proposals are learned instead of handcrafted.

## See Also
- [[Anchors]]
- [[Window Centers]]
