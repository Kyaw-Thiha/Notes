# Tactical Network with CTDE

## Per-unit Policy Network

```
o_i = {LocalCrop, SelfFeat, NearbyEnemies, GoalToken, ManeuverToken, Time}
   ↓
[HexConv / GNN Encoder]
   ↓
[Memory: GTrXL / GRU]
   ↓
[Action Heads + Masks]
   ├─ MoveDirHead ⊙ mask_move
   ├─ ModeHead    ⊙ mask_mode
   └─ TargetPointer ⊙ mask_target
```

- **Outputs:** π_L^i(a_i | o_i, z_g, z_m), value estimate v_i.  
- **Shared weights across all units.**  
- **Action masking:** invalid moves receive logit = –∞.

---

## Centralized Critic

```
[AttnPool across friendly units]
     + EnemySummary
     + GlobalMapFeat
     + Goal + Maneuver tokens
       ↓
   V_C(s; g, m)   # joint value
```

- Used **only during training** for PPO/HAPPO advantage estimates.  
- At execution, each unit acts independently.
