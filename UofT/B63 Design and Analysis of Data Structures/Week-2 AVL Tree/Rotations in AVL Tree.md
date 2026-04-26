# Rotations in AVL Tree
There are $4$ main basic rotations for [[AVL Tree]]:
- [[#Left Rotation]]
- [[#Right Rotation]]
- [[#Left-Right Rotation]]
- [[#Right-Left Rotation]]

---
### Left Rotation
> A single rotation of counter clockwise.
> Used with balance values of $-2, -1, \ 0$.

Consider the tree below:
![image|200](https://notes-media.kthiha.com/Rotations-in-AVL-Tree/fec037a43eed8a9a6ab00b3ced1b3680.png)

If we rotate counter clockwise about $A$, we get
![image|200](https://notes-media.kthiha.com/Rotations-in-AVL-Tree/b6aff28c2a1e7d3780b4c3e3fab2ceaa.png)

---
### Right Rotation
> A single rotation of clockwise.
> Used with balance values of $2, \ 1, \ 0$.

Consider the tree below:
![image|200](https://notes-media.kthiha.com/Rotations-in-AVL-Tree/1fc997cd7da352d182c525aa385b58ad.png)

If we rotate clockwise about $A$, then we get:
![image|200](https://notes-media.kthiha.com/Rotations-in-AVL-Tree/ddec344b4e1ca210a99b12e77080cb8f.png)

---
### Left-Right Rotation
> A double rotation of counter clockwise, then clockwise.
> Used with balance values of $-2, \ 1, \  0$ or $2, -1, \ 0$.

Consider the tree below:
![image|200](https://notes-media.kthiha.com/Rotations-in-AVL-Tree/6f032346cace4a6326f5366e13ad6081.png)

If we rotate counter clockwise about $A$, then we get
![image|200](https://notes-media.kthiha.com/Rotations-in-AVL-Tree/698b0c5d0d1964ee15c161565e25d11b.png)

Then if we rotate clockwise about $C$, we get
![image|300](https://notes-media.kthiha.com/Rotations-in-AVL-Tree/6aad683a54e97b2cf165ba3655a8cbe9.png)

---
### Right-Left Rotation
> A double rotation of clockwise, then counter clockwise.
> Used with balance values of $-2, \ 1, \  0$ or $2, -1, \ 0$.

Consider the tree below:
![image|300](https://notes-media.kthiha.com/Rotations-in-AVL-Tree/171102dc687b0ef2fd13301a85516741.png)

---
## See Also
- [[AVL Tree]]
- [[AVL Tree Height]]
- [[AVL Tree Algorithms]]
- [[Union of AVL Trees]]
- [[Intersection of AVL Trees]]
- [[Difference of AVL Trees]]
