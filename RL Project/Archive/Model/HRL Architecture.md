# HRL Architecture

## Hierarchy Levels

### Strategic Policy (π_H)
- **Input:** global map summary, encoded goals, time/score.  
- **Output:** subgoal g = {goal_type, target (q,r), time_budget, priority}.  
- **Cadence:** every 6–12 turns.  
- **Optimization:** actor-critic, sparse rewards (VP, objective captures).

---

### Operational Policy (π_M)
- **Input:** global features + subgoal token (z_g).  
- **Output:** maneuver m = {type: flank, breakthrough, hold, etc., AO polygon}.  
- **Cadence:** every 2–4 turns.  
- **Optimization:** actor-critic, rewards for subgoal completion inside AO.

---

### Tactical Controllers (π_L^i)
- **Input:** per-unit local crop, self features, nearby enemies, goal/maneuver tokens.  
- **Output:** discrete actions (move, mode, target) with action masks.  
- **Shared policy across units.**

---

## Centralized Critic (V_C)
- **Sees all units + global state during training.**
- **Computes joint value estimate** for advantage calculation.  
- **Execution time:** each unit acts independently (decentralized).
