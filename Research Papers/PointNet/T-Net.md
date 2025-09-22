# T-Net
#cv/3d/object-detection/point-net #cv/3d/object-segmentation/point-net

`T-Net` are mini [[PointNet]] that are used to make model invariant to
- order of input points
- specific object transformations like rotation & translation

## Architecture
### First T-Net
The `First T-Net` is used to make `PointNet` invariant to specific object transformations.
It take $3 \times n$ 3D points, and output $3 \times 3$ affine transformation matrix.

```D
Input: N × 3 point cloud
        │
        ▼
  ┌─────────────── Shared MLP ────────────────┐
  │  3 → 64 → 128 → 1024  (BN+ReLU each)      │
  └───────────────────────────────────────────┘
        │
        ▼
   Max Pool across N points
        │
        ▼
 Global feature [1024]
        │
        ▼
  FC 512 → FC 256 → FC 9
   (BN+ReLU)   (BN+ReLU)   (no BN/ReLU)
        │
        ▼
 Reshape → 3×3 matrix
   (init = Identity)
        │
        ▼
 Transform input points
```

The main key points here are
- `Shared MLP`
- `Max Pooling` which is a symmetric function

## Second T-Net
The `Second T-Net` is used to make `PointNet` invariant to points input order.
In takes $64 \times N$ input features, and output $64 \times 64$ matrix

```D
Input: N × 64 point features
        │
        ▼
  ┌─────────────── Shared MLP ────────────────┐
  │  64 → 64 → 128 → 1024  (BN+ReLU each)     │
  └───────────────────────────────────────────┘
        │
        ▼
   Max Pool across N points
        │
        ▼
 Global feature [1024]
        │
        ▼
  FC 512 → FC 256 → FC 4096
   (BN+ReLU)   (BN+ReLU)   (no BN/ReLU)
        │
        ▼
 Reshape → 64×64 matrix
   (init = Identity)
        │
        ▼
 Transform point features
```

Compared to the `First T-Net`, forming a transformation matrix on higher dimension ($64 \times 64$ compared to $3 \times 3$) is much harder.

So, a `L2 regularization term` is added to the `softmax training loss`.
$$
L_{reg} = || I - A.A^T ||_{F}^2
$$
where $A$ is the feature alignment matrix

This helps the transformation matrix $A$ to be close to an `orthogonal matrix`.


#### Why do we want to be close to orthogonal?

Being `orthogonal` means lengths & angles are preserved.
So, there is only rotation & translation.
But no skewing, stretching, and no collapsing to smaller dimension.

## MLP vs Shared MLP
### Why do we need Shared-MLP?
If we use `normal MLP`, 
- Input: $[p_{1}, p_{2}, p_{3}]$
  Output: $[MLP_{1}(p_{1}), MLP_{2}(p_{2}), MLP_{3}(p_{3})]$
- Input: $[p_{2}, p_{1}, p_{3}]$
  Output: $[h_{1}(p_{2}), h_{2}(p_{1}), h_{3}(p_{3})]$
where $h_{n}$ stands for each `MLP`

Since the `MLP` is updated for each point (or batch of them), the order of these points will matter on the output features.

If we use `shared MLP` (or `1x1 Conv`),
- Input: $[p_{1}, p_{2}, p_{3}]$
- Output: $[h(p_{1}), h(p_{2}), h(p_{3})]$
This is just permutation of the set, so order of input does not matter.

### MLP Code
```python
class NormalMLP(nn.Module):
    def __init__(self, in_dim=3, hidden=(64, 128), out_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], out_dim)

    def forward(self, x):
        # x: [B, in_dim]
        x = F.relu(self.fc1(x))     # [B, 64]
        x = F.relu(self.fc2(x))     # [B, 128]
        x = self.fc3(x)             # [B, out_dim]
        return x
```

### Shared MLP 
```python
class SharedMLP_Linear(nn.Module):
    def __init__(self, in_dim=3, hidden=(64, 128, 1024)):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], hidden[2])

    def forward(self, x):
        # x: [B, N, in_dim]
        B, N, D = x.shape
        x = x.view(B*N, D)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(B, N, -1)        # [B, N, 1024]
        return x
```

### Shared MLP (1x1 Conv)
```python
class SharedMLP_Conv1d(nn.Module):
    def __init__(self, in_ch=3, hidden=(64, 128, 1024)):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, hidden[0], kernel_size=1)
        self.conv2 = nn.Conv1d(hidden[0], hidden[1], kernel_size=1)
        self.conv3 = nn.Conv1d(hidden[1], hidden[2], kernel_size=1)

    def forward(self, x):
        # x: [B, C_in, N]
        x = F.relu(self.conv1(x))   # [B, 64, N]
        x = F.relu(self.conv2(x))   # [B, 128, N]
        x = self.conv3(x)           # [B, 1024, N]
        return x
```
