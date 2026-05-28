# Reinforcement Learning Lab

A Python reinforcement learning laboratory containing model-free agents (tabular Q-Learning and SARSA) designed to solve pathfinding tasks in discrete grid worlds, built from scratch using NumPy.

## Project Structure
```
Reinforcement-Learning-Lab/
├── gridworld.py       # Custom Gym-style discrete grid environment
├── q_learning.py      # Q-learning and SARSA tabular update runners
├── requirements.txt   # Core Python dependencies
└── README.md          # Setup and reinforcement boundaries guide
```

## Setup & Running

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run training simulations (runs Q-learning updates over 1000 episodes on a 5x5 Gridworld grid, then exports convergence statistics):
   ```bash
   python q_learning.py
   ```
3. The script will save `reward_curve.png` and output the learned Q-value grid in your terminal console.

## Environment Details
- **Grid Layout:** 5x5 grid cells. State is coordinate tuple `(x, y)`.
- **States:** 
  - Start: `(0, 0)`
  - Goal: `(4, 4)` (Reward: `+10`)
  - Pit: `(2, 2)` (Reward: `-10`)
- **Actions:** Discrete directional moves: Up (0), Down (1), Left (2), Right (3).
- **Decay:** $\epsilon$-greedy exploration decaying exponentially from 1.0 to 0.05.
