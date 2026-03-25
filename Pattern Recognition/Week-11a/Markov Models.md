# Generative Sequences
Suppose we want to deal with sequences $(x_{1}, \ \dots, \ x_{T})$.
We would model the distribution as
$$
p(x_{1}, \ \dots,\ x_{T}) 
= \prod^{T}_{t=1} p(x_{t} \mid x_{1}, \ \dots, \ x_{t-1})
$$
But carrying that history means the complexity of the distribution of the latest sample grows with the sequence history.

We can set up an $n^{th}\text{-order}$ `Markov model`: 
$$
\begin{align}
p(x_{t} \mid x_{1}, \ \dots, \ x_{t-1}) 
&= p(x_{t} \mid x_{t-1}) & & \text{First Order} 
\\[6pt]

p(x_{t} \mid x_{1}, \ \dots, \ x_{t-1}) 
&= p(x_{t} \mid x_{t-1}, \ x_{t-2})  
& & \text{Second Order} \\[6pt]
\end{align}
$$
This makes solving for these models to be much simpler

---
## Solving for Markov Model
Consider the $1^{st}\text{-order}$ markov model.
Then, the distribution we are trying to learn becomes:
$$
p(x_{1}, \ \dots, \ x_{T})
= p(x_{1}) \ \prod^{T}_{t=2} p(x_{t} \mid x_{t=1})
$$
and our [[Risk Function|risk function]] becomes:
$$
\mathbb{E}_{S}[-\log( \ p(x_{t} \mid x_{t-1}; \ \theta) \ )]
$$
where we can take our sequence data and construct the dataset:
$$
S=( \ (x_{1}, \ y_{1}=x_{2}), 
\ (x_{2}, \ y_{2}=x_{3}), \ \dots, 
\ (x_{T-1}, y_{T-1} = X_{T}) \ )
$$
All we need is something that can learn the appropriate probability distribution. Then, we can run our usual optimization techniques on the [[Risk Function|risk function]].

---
