# Convolutional CVAE
Compared to [[Conditional Variational AutoEncoders|CVAE]], fully-connected dense neural networks flatten the image and thus, lose the structure.
[[Convolution Layer]] can preserve local structures like edges  and curves, so it result in sharper reconstructions.

## Encoder
Here is a typical `Conv-CVAE Encoder`.
```
Input: (1, 28, 28) digit image + condition y
   │
   ├─ Conv2D(32 filters, 3x3) + ReLU
   ├─ Conv2D(64 filters, 3x3) + ReLU
   ├─ Flatten
   └─ Dense → outputs μ(x,y), logσ²(x,y)
```

## Decoder
Here is a typical `Conv-CVAE Decoder`.
```
Input: z + y  (latent vector + condition)
   │
   ├─ Dense → reshape to feature map (e.g., (64, 7, 7))
   ├─ ConvTranspose2D(64 filters, 3x3) + ReLU
   ├─ ConvTranspose2D(32 filters, 3x3) + ReLU
   └─ ConvTranspose2D(1 filter, 3x3, activation=sigmoid)
        → Output: (1, 28, 28) reconstructed image
```

## See Also
- [[Conditional Variational AutoEncoders]]
- [[Variational AutoEncoders]]
- [[AutoEncoders]]
- [[Convolution Layer]]
