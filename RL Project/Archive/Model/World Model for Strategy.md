# World Model for Strategy

**Purpose:** accelerate strategic learning by imagining goal sequences without real environment steps.

---

## Components

```
[RSSM World Model]
   Encode(GlobalFeat_t) -> z_t
   (s_t, a_H,t) -> s_{t+1}
   Decoder predicts: map summary, VP, supply proxies
```

---

## High-Level Policy (π_H)
- Trains on **imagined rollouts** of the world model.
- Chooses subgoals in latent space.
- Optimized with λ-returns over imagined trajectories.

---

## Low-Level Policy (π_L)
- Still trained on **real environment rollouts** with PPO/HAPPO.

---

## When to use
- Simulator is slow.  
- Strategic horizon is very long.  
- You want to test goal sequences without playing entire games.
