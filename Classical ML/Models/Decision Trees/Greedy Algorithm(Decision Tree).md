# Greedy Algorithm
#ml/models/classic/decision-tree/greedy-algorithm 

This is a recursive algorithm that grow the tree from node, and build the tree one node at a time.
```python
for each feature 𝑙
	for each threshold 𝜏
		compute InfoGain(D_j, l, t)

Select the split function with max InfoGain

If stopping criteria is met
	stop splitting node
```

where stopping criteria can be
- Depth of tree
- Entropy Threshold
- Minimum number of points in node

---
## See Also
- [[Classical ML/Models/Decision Trees/Decision Tree]]
- [[Information Gain]]
