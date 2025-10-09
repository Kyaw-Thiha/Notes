# Deep Q-Learning (DQN)
#rl/algorithms/dqn

Note that computing the `Q-Table` directly for an environment with big value of `State Space` & `Action Space` is inefficient.

So, `DQN` solves this by using neural networks to estimate the optimal `Q-Table`.

![DQN](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit4/deep-q-network.jpg)

When preprocessing a `Continuous Action/State Space`  like an `Atari Game`, we usually 
- Convert it to Black-White (1 Channel) 
- Downsample the resolution
- Stack $n$ number of frames together to help the model learn motion

## Objective Function
Recall that in [[Q-Learning]], we train by updating the `Q-table` using 
$$
Q(S_{t}, A_{t})  = Q(S_{t}, A_{t}) + \alpha.[R_{t+1} + \gamma.max_{a}Q(S_{t+1}, a) - Q(S_{t}, A_{t})] \\
$$

Hence, the [[Temporal Difference]] Target is 
$$
y_{t} = R_{t+1} + \gamma.max_{a}Q(S_{t+1}, a)
$$

We need to minimize
$$
\delta_{1} = y_{1} - Q(S_{t}, A_{t}, \theta)
$$

So, the [[Loss Function]] is 
$$
L(\theta) = E[(y_{t} - Q(S_{t}, A_{t}; \theta))^2]
$$
To update the `parameters`, we perform a `gradient descent`
$$
\begin{aligned}
L(\theta) &= \tfrac{1}{2}\,(y_t - Q(S_t, A_t; \theta))^2 
&\text{(TD loss)} \\[6pt]

\nabla_\theta L(\theta) &= - (y_t - Q(S_t, A_t; \theta)) \, \nabla_\theta Q(S_t, A_t; \theta)
= -\delta_t \, \nabla_\theta Q(S_t, A_t; \theta)
&\text{(gradient of loss)} \\[6pt]

\theta &\leftarrow \theta - \alpha \nabla_\theta L(\theta)
&\text{(gradient descent step)} \\[6pt]

\theta &\leftarrow \theta - \alpha (-\delta_t \, \nabla_\theta Q)
&\text{(substitute gradient)} \\[6pt]

\theta &\leftarrow \theta + \alpha \, \delta_t \, \nabla_\theta Q(S_t, A_t; \theta)
&\text{(final update rule)}
\end{aligned}
$$


## Stabilization
Due to using non-linear functions in a neural network, and bootstrapping ([[Temporal Difference]]), `DQN` may suffer from instability.

There are different ways to solve this

### Experience Replay
In `Experience Replay`, we use a `replay buffer` to store tuples of experience ($s_{t}, a_{t}, r_{t}, s_{t+1}$), and then sample a small subset out of it for training.
It allows the network to have a sort of long-term memory.
We can define the capacity of `replay buffer` as a hyperparameter.

- Random sampling remove correlation in `observation` sequences, and avoid `action values` from oscillating or diverging.
- It prevents `catastrophic forgetting`, whereby the network forget the previous experiences as it gets new experiences.
- It allows more `efficient` use of experiences.

### Fixed Q-Target
Note that since we are using [[Temporal Difference]], we can only use [[Bellman Equation]] to estimate the `Q-Target`.
So, updating the `Q-Target` as we train will lead to the network never converging (think of both predictions & targets moving in same direction).
So, we solve this by fixing the `Q-Target`.

### Double DQN
Since we are using [[Temporal Difference]], we only have an estimate of `Q-Target`.
This means that we cannot be sure that the best `action` for the next state is the action with the highest `Q-value`.

So, we solve this by using 2 networks to decouple `action selection` from `target generation`.
- Our `DQN Network` is used to select the best action to take for next state. (action with higest `Q-Value`)
- Our `Target Network` is used to estimate the target `Q-Value` of taking that action.

Note that `DQN Online Network` is updated every steps, while `Target Network` gets updated periodically (every $N$ steps, with maybe soft updating).

`Double DQN` help reduce overestimation of `Q-Values`, and lead to more stable training.

### Other methods
There are also other methods like
- `Prioritized Experience Replay`
- `Dueling Deep Q-Learning`

## See Also
- [[Q-Learning]]
- [[Temporal Difference]]


