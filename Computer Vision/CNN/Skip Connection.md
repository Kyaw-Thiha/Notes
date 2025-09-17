# Skip Connection
#cv/cnn/skip-connection #cv/cnn/residual  
Skip Connections (Residuals) can be used to help the model stabilize the training.

![Skip Connection](https://www.researchgate.net/profile/Peng-Yi-13/publication/329750500/figure/fig1/AS:706193672650755@1545381100642/Different-skip-connection-schemes-a-No-skip-connection-b-Distinct-source-skip.ppm)

Instead of seeing it as adding a new term, think of it this way - 

If the actual learning required is $F(x) = H(x) + b$, 
now the model only needs to learn $H(x)$, while $b$ is added from the past layers.

This is automatically taken into consideration during back-propagation process.

Also, note that it is a piece-wise addition process; not a concatenation (some models does concatenation).

