An MDP consists of 
1. A set of states: $s \in \mathcal{S}$
2. A set of actions: $a \in \mathcal{A}_{S}$
3. A set of rewards: $r \in \mathcal{R}$
4. Problem Dynamics: $Pr(s_{t+1} = s', r_{t}=r \mid s_{t}=s, a_{t}=a)$

Objective: Learn a policy $\pi(\mathbf{a} \mid \mathbf{s}) = P(A_{t}=\mathbf{a} \mid S_{t} =s)$
such that the return $G_{t}=\sum ^{\infty}_{t} \gamma^{t} R_{t}$ is maximized, $\gamma \in ]0,1[$.