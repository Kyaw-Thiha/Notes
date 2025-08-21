# HRL + CTDE Overview

**Core idea:**  
Train an agent with **Hierarchical Reinforcement Learning (HRL)** on a 25×25 hex wargame map, where:
- **High-level (Strategic)** chooses subgoals (cities, hills, river lines).
- **Mid-level (Operational)** translates subgoals into maneuvers (flank, breakthrough, hold).
- **Low-level (Tactical)** executes unit actions per turn.  
At the tactical layer, use **Centralized Training, Decentralized Execution (CTDE)** so individual unit policies coordinate effectively.

---

## Why HRL + CTDE?
- **Aligns with layered goals** (strategy → ops → tactics).
- **Scales to larger maps** (25×25).
- **Supports both preset and dynamic goals.**
- **CTDE** resolves credit assignment when many units must coordinate.

---

## Extensions
- [[Short-Horizon Search (PUCT)]] — sharpen tactical moves.
- [[World Model for Strategy]] — strategic imagination in latent space.
- [[VIN Planner]] — movement/supply-aware tactical planning.
