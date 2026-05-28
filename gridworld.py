import numpy as np

class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.state = (0, 0)
        self.goal = (size - 1, size - 1)
        self.pit = (size // 2, size // 2)
        
        # Action mappings: 0=Up, 1=Down, 2=Left, 3=Right
        self.action_space = [0, 1, 2, 3]
        self.num_states = size * size
        
    def reset(self):
        self.state = (0, 0)
        return self.state_to_index(self.state)
        
    def state_to_index(self, state):
        return state[0] * self.size + state[1]
        
    def step(self, action):
        x, y = self.state
        
        if action == 0:   # Up
            x = max(0, x - 1)
        elif action == 1: # Down
            x = min(self.size - 1, x + 1)
        elif action == 2: # Left
            y = max(0, y - 1)
        elif action == 3: # Right
            y = min(self.size - 1, y + 1)
            
        self.state = (x, y)
        state_idx = self.state_to_index(self.state)
        
        # Determine reward and termination
        if self.state == self.goal:
            reward = 10.0
            done = True
        elif self.state == self.pit:
            reward = -10.0
            done = True
        else:
            reward = -1.0 # step penalty to encourage shortest path
            done = False
            
        return state_idx, reward, done
