# Short-Horizon Search (PUCT)

**Purpose:** sharpen tactical exchanges and move ordering in contact areas.

---

## Flow

```
Local Battle Subgraph S_local (units within radius R of enemy)
   ↓
[PUCT Search]
   - Priors: tactical policy π_L
   - Masks: applied at each node
   - Leaf eval: centralized critic V_C
   - Budget: 64–128 sims, depth 2–3
   ↓
Improved policy π̂
   ↓
Execute / Train
   - Execute actions from π̂
   - Train π_L to match π̂ (policy distillation)
```

---

## When to use
- After tactical baseline (HRL + CTDE) is competent.  
- Adds 2–6× compute on contact turns, but much better micro-tactics.
