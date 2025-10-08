# Components of RL
#rl

This RL loop outputs a sequence of `state`, `action`, `reward` and `next state`.

![Components](https://cas-bridge.xethub.hf.co/xet-bridge-us/637f72fc62a4445929f4fcb3/1af1a305e6c7f1459164e307b6b8e67cc1ccbe5c35cb23a3c0136b939256714a?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=cas%2F20251008%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20251008T021745Z&X-Amz-Expires=3600&X-Amz-Signature=b64c510b38e36f9ca3c18cfff1aae3c1af843cbf130e4ab3da7374358bff5644&X-Amz-SignedHeaders=host&X-Xet-Cas-Uid=65087834a2abcb18d60bfbc0&response-content-disposition=inline%3B+filename*%3DUTF-8%27%27RL_process_game.jpg%3B+filename%3D%22RL_process_game.jpg%22%3B&response-content-type=image%2Fjpeg&x-id=GetObject&Expires=1759893465&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc1OTg5MzQ2NX19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2FzLWJyaWRnZS54ZXRodWIuaGYuY28veGV0LWJyaWRnZS11cy82MzdmNzJmYzYyYTQ0NDU5MjlmNGZjYjMvMWFmMWEzMDVlNmM3ZjE0NTkxNjRlMzA3YjZiOGU2N2NjMWNjYmU1YzM1Y2IyM2EzYzAxMzZiOTM5MjU2NzE0YSoifV19&Signature=IRmNJkNtlg2KY7OtTeO99Rimf5%7E6TCdQTbj0NV3qDIQmiV6WBZ47Rz0Lgm2xj7sXuou0Hgs8EGjKkMD22x-l0XMGFNUnlmu0QA9uSvqySScHsQ7x0iyHAKQcdPAcQYsPnogXuwhQ-KDvJ6Farw8J9zTtEzBmfpRCWscOosZI6WBViqJ327fuI9kqNSoXQWEGwk9Rjo8y2P-2vXZX-xYq%7Eg7Nz4qUGgd7YCD6RTsfBwPAxIVtrOf2PDwNvjSB7AQSB2Aj7CyJGdbb0dINoXammvGLuKhn1keheNOnGAW5pMZzT-r-ArwLfIZfIWf26bMm1CWjgrLZAXzr7Mug%7Ezq-BA__&Key-Pair-Id=K2L8F4GPSG1IFC)

1. The agent receives state $S_{t}$ from the `environment`.
2. Based on state $S_{t}$, the agent take action $A_{0}$.
3. The environment goes to new state $S_{t+1}$.
4. The environment gives some reward $R_{1}$ to the agent.

The agent’s goal is to maximize its cumulative reward, called the `expected return`.

## Observation Space

`State` $S$: complete description of the state of world
`Observation` $o$: partial description of the state

![[Observation Space.png]]

## Action Space
The `Action Space` is the set of all possible actions in the envrionment.

`Discrete Action Space`: Number of possible actions is finite.
`Continuous Action Space`: Number of possible actions is infinite.

![[Discrete vs Continous.png]]

## Reward
The `cumulative reward function` is used to give feedback to the agent about its actions.

![[Reward.png]]

First, let's think of a trivial case of cumulative reward.
$$
R(\tau) = r_{t+1} + r_{t+2} + r_{t+3} + \dots
$$

However in reality, the rewards at the start are more likely to happen than those later.
So, we should discount the rewards further away with a weight called gamma.

$$
R(\tau) = r_{t+1} + \gamma.r_{t+2} + \gamma^2.r_{t+3} + \dots
$$

where $0.95 < \gamma < 0.99$.

## Types of Tasks

- `Episodic Tasks`: Tasks that have starting point & ending points.
- `Continuing Tasks`: Tasks that continue forever.

![[Episodic vs Continuing.png]]

## Exploration vs Exploitation
- `Exploration`: Trying random actions to find more information about the environment.
- `Exploitation`: Exploiting known information to maximize rewards.

![[Exploration vs Exploitation.png]]
