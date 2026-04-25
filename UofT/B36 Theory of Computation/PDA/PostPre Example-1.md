# PostPre Example-1

`Question`
Let $L_{2a} = \{ x \in \Sigma^*: \#_{00}(x) = \#_{11}(x) + 1 \}$

`Strategies`
Start with `D-PDA` for $L_{2}$
Then, you have two options 
- `preprocess`
- `post-process`

`Preprocess`
This is preferred by Nick since it allows deterministic to be left behind.
![[Preprocess.png]]

`PostProcess`
![[PostProcess.png]]
