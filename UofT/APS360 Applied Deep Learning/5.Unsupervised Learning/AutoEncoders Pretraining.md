# AutoEncoder Pretraining
[[AutoEncoders Pretraining|Autoencoder pretraining]] is a [[Self-Supervised Learning|self-supervised learning technique]] where 
- an [[AutoEncoders|encoder-decoder]] is first trained to reconstruct unlabelled input data
- then its weights are transferred to downstream tasks

![image|400](https://notes-media.kthiha.com/AutoEncoders-Pretraining/52aba27935a58f67870ede94286bfb3e.png)

It essentially
- reconstruct the input
- by computing loss in output space
- and compressing all details into latent space

---
## See Also
- [[Self-Supervised Learning]]
- [[AutoEncoders]]
- [[Contrastive Learning]]
- [[Transfer Learning]]
