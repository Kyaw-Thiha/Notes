# Weight Balanced Tree Rotations
In general there are $4$ cases to look for.

**Case-1**: Left heavy, left heavy
- For this case, $V$ is left heavy and when we let $X = \text{V.left}$, $X$ is also left heavy.
- Here, we rotate clockwise about $V$.
- Look at [[#Example-3]].

**Case-2**: Left heavy, right heavy
- For this case, $V$ is left heavy and when we let $X = V.left$, $x$ is right heavy.
- Here, we rotate counter clockwise about $X$ and then clockwise about $V$.
- Look at [[#Example-4]].


**Case-3**: Right heavy, right heavy
- For this case, $V$ is right heavy and when we let $X = V.right$, $X$ is also right heavy.
- Here, we rotate counter clockwise about $V$.
- Look at [[#Example-1]].

**Case-4**: Right heavy, left heavy
- For this case, $V$ is right heavy and when we let $X = V.right$, $X$ is left heavy.
- Here, we rotate clockwise about $X$, then counter clockwise about $V$.
- Look at [[#Example-2]].

---
## General Algorithm for Balancing
Here is the general algorithm for balancing [[Weight-Balanced Trees|WBTs]].
![image|350](https://notes-media.kthiha.com/Weight-Balanced-Tree-Rotations/6fb15dff8bae12a9aca55070700e7074.png)

---
### Example-1
**Question**
![image|300](https://notes-media.kthiha.com/Weight-Balanced-Tree-Rotations/4a855a75d56ed47d5e1818ac3383f74d.png)

**Solution**
![image|300](https://notes-media.kthiha.com/Weight-Balanced-Tree-Rotations/1e3a273a57b0258e1bce3916b57b4667.png)
![image|300](https://notes-media.kthiha.com/Weight-Balanced-Tree-Rotations/a5c5f90d39f4c200426307e9c28193dd.png)

---
### Example-2
**Question**
![image|300](https://notes-media.kthiha.com/Weight-Balanced-Tree-Rotations/dfdb830819ce0a03ca399054ede9deff.png)

**Solution**
![image|300](https://notes-media.kthiha.com/Weight-Balanced-Tree-Rotations/ea176bbfb3fb3ae93e64b843707e8f22.png)
![image|300](https://notes-media.kthiha.com/Weight-Balanced-Tree-Rotations/7613bb7c57eb218fa918f79b41089d4e.png)

---
### Example-3
**Question**
![image|300](https://notes-media.kthiha.com/Weight-Balanced-Tree-Rotations/585a8c11bb4f1f88ccb34b4145defe75.png)

**Solution**
![image|300](https://notes-media.kthiha.com/Weight-Balanced-Tree-Rotations/fee221c35718633ab193a0fb40f06e40.png)

---
### Example-4
**Question**
![image|300](https://notes-media.kthiha.com/Weight-Balanced-Tree-Rotations/df0317a91a21c0f815b7206e4ba5fdbc.png)

**Solution**
![image|300](https://notes-media.kthiha.com/Weight-Balanced-Tree-Rotations/21814c81f9e323caa9b44b9ee75e1b27.png)
![image|300](https://notes-media.kthiha.com/Weight-Balanced-Tree-Rotations/df7a850fcfe09087ddccb007aa4c0502.png)

---
## See Also
- [[Weight-Balanced Trees]]
- [[Interval Tree]]
- [[Augmented AVL Tree]]
- [[Augmented Data Structures]]
