# Q-Learning
#rl/algorithms/q-learning

`Q-Learning` is an [[Off-Policy Learning|Off-Policy]] [[Value-Based Methods|Value-Based]] method that uses a [[Temporal Difference]] approach train its `action-value function`.

![Q-Learning](https://cas-bridge.xethub.hf.co/xet-bridge-us/637f72fc62a4445929f4fcb3/1110b1b4f0a1f064d2f112b4d0eb88dc2a478e057be94fbe8036d6a534d11236?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=cas%2F20251008%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20251008T232822Z&X-Amz-Expires=3600&X-Amz-Signature=501880ba2d205df2894063b45546103a11055409b36763ed01beef6298e30de5&X-Amz-SignedHeaders=host&X-Xet-Cas-Uid=65087834a2abcb18d60bfbc0&response-content-disposition=inline%3B+filename*%3DUTF-8%27%27Q-function-2.jpg%3B+filename%3D%22Q-function-2.jpg%22%3B&response-content-type=image%2Fjpeg&x-id=GetObject&Expires=1759969702&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc1OTk2OTcwMn19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2FzLWJyaWRnZS54ZXRodWIuaGYuY28veGV0LWJyaWRnZS11cy82MzdmNzJmYzYyYTQ0NDU5MjlmNGZjYjMvMTExMGIxYjRmMGExZjA2NGQyZjExMmI0ZDBlYjg4ZGMyYTQ3OGUwNTdiZTk0ZmJlODAzNmQ2YTUzNGQxMTIzNioifV19&Signature=iCsv6rv07rezj06diXT1aWvE2otKU0ADShEZMHaDH4zhU%7ER24BIr2840orA0Su5CQcHcQUdvL%7EFTrlrTyFJMK%7EL4kL0IiD90GAQnBe37gSyrrMzDMK1cJc2h4nc%7EJgSr1v3eSTECkHO9P7xeVsSqZIWnPnq8zdFBQS-K2aGFs8dbX7bbzopT6k9eP0BlrxfMjE2TTO%7EhMu-1QHnuQ3afbz7ZOiwjxcllAtPiOJ4eYtwrXRdowisCYgXMRqKqZc9771Fov9JKWhqsdf7CR-GfCvKH4oE7CoKNZ3JFJkvh6-QhiNBqWkfogZGf%7EnKfwFH4HUmbctag1i9ue8qxV6o7vA__&Key-Pair-Id=K2L8F4GPSG1IFC)

## Training Q-Learning

![Q-Learning Training](https://huggingface.co/blog/assets/73_deep_rl_q_part2/Q-learning-1.jpg)

1. We initialize the `Q-Table` for each state-action pair.
2. Choose an action based on [[Epsilon Greedy Strategy]].
3. Based on `action` $A_{t}$, we get `reward` $R_{t+1}$ and `state` $S_{t+1}$
4. Since this is [[Temporal Difference]] learning, we update $Q(S_{t}, A_{t})$ every step.

We calculate the `TD Target` using [[Greedy Strategy]].
$$
y_{t} = R_{t+1} + \gamma.max_{a}Q(S_{t+1}, a)
$$

Then, we use that `TD-Target` to update the policy function.

$$
\begin{align}
Q(S_{t}, A_{t})  
&= Q(S_{t}, A_{t}) + \alpha.[y_{t} - Q(S_{t}, A_{t})] \\[6pt]
&= Q(S_{t}, A_{t}) + \alpha.[R_{t+1} + \gamma.max_{a}Q(S_{t+1}, a) - Q(S_{t}, A_{t})] \\
\end{align}
$$

## See Also
- [[On-Policy Learning]]
- [[Value-Based Methods]]
- [[Temporal Difference]]
- [[SARSA]]

