# VIN Planner (Value-Iteration Network)

**Purpose:** improve tactical routing, supply movement, and terrain-aware decisions.

---

## Architecture

```
LocalCrop_i -> Conv features -> Cost map C
   ↓
[VIN Block]
   K iterations of (Conv + max)  # Bellman backups
   ↓
Q_move (scores for 6 directions)
```

- Concatenate Q_move with other action head features.
- Apply PPO/HAPPO training as usual (with masks).

---

## Advantages
- Differentiable planner learns terrain/supply costs.  
- Scales well with dilated convs for larger receptive fields.  
- Especially useful when poor routing is the main failure.
