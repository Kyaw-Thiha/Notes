# Proximal Policy Optimization

Compared to other [[Policy-Based Methods]], `PPO` focus on improving the training stability by preventing large update on the `policy function` $\pi_{\theta}(s_{t})$.

![PPO](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit9/cliff.jpg)

It could use techniques like `clipping` and `KL Divergence Control`.

The `ratio function` can be used to represent the updates to the policy function $\pi_{\theta}(a_{t} | s_{t})$ 
$$
r_{t}(\theta) = \frac{\pi_{\theta} (a_{t} | s_{t})}{\pi_{\theta_{old}}(a_{t} | s_{t})}
$$

This means
- If $r_{t}(\theta)$ > 1: the `action` $a_{t}$ is more likely in new `policy`
- If $0 > r_{t}(\theta) > 1$: the `action` $a_{t}$ is less likely in new `policy`

To constrain this `ratio function`,
$$
min(\ r_{t}(\theta). A_{t},\ clip(r_{t}(\theta),\ 1-\epsilon,\ 1+\epsilon ).A_{t} \ )
$$
where 
- $A_{t}$ is the [[Actor-Critic Methods#Advantage Function|Advantage Function]]
- $\epsilon$ is a hyper-parameter we can set

## Clipped Surrogate Objective Function

Hence, we can define the `clipped surrogate objective function` as
$$
L^{CLIP}(\theta) = E_{t} [\ min(\ r_{t}(\theta). A_{t},\ clip(r_{t}(\theta),\ 1-\epsilon,\ 1+\epsilon ).A_{t} \ ) \ ]
$$
where 
- $E_{t}[f_{t}] = \frac{1}{N} \sum^N_{t=1} f_{t}$  is the `expected value`

## Objective Function

Combining this, we get our `PPO Objective Function`,

$$
L^{PPO}(\theta) = 
E_{t}[L^{CLIP}(\theta)] 
+ c_{1}.L^{VF}(\theta) 
- c_{2}.S[\pi_{\theta}](s_{t})
$$

- $L^{CLIP}(\theta)$: `Clipped Surrogate Objective Function`
- $L^{VF}$: `Value Function Loss` (Mean Squared Error)
- $S[\pi_{\theta}]$: `Entropy Bonus` (Encourage exploration)
- $c_{1}$, $c_{2}$: Weighing coefficient `hyperparameters`
