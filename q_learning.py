import numpy as np
import matplotlib.pyplot as plt
from gridworld import GridWorld

def epsilon_greedy_policy(q_table, state, epsilon, action_space):
    if np.random.rand() < epsilon:
        return np.random.choice(action_space) # Explore
    else:
        return np.argmax(q_table[state]) # Exploit

def run_q_learning():
    env = GridWorld(size=5)
    
    # Initialize Q-table: states x actions
    q_table = np.zeros((env.num_states, len(env.action_space)))
    
    # Hyperparameters
    alpha = 0.1
    gamma = 0.95
    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.05
    episodes = 1000
    
    rewards_history = []
    
    print(f"Training Q-Learning agent for {episodes} episodes...")
    
    for ep in range(episodes):
        state = env.reset()
        done = False
        cumulative_reward = 0
        
        while not done:
            action = epsilon_greedy_policy(q_table, state, epsilon, env.action_space)
            next_state, reward, done = env.step(action)
            
            # Bellman update rule: Q(s,a) = Q(s,a) + alpha * [r + gamma*max Q(s',a') - Q(s,a)]
            best_next_action = np.argmax(q_table[next_state])
            td_target = reward + gamma * q_table[next_state][best_next_action]
            q_table[state][action] += alpha * (td_target - q_table[state][action])
            
            state = next_state
            cumulative_reward += reward
            
        # Decay exploration rate
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        rewards_history.append(cumulative_reward)
        
        if (ep + 1) % 200 == 0:
            print(f"Episode {ep+1:03d}/{episodes:03d} | Epsilon: {epsilon:.3f} | Last Cumulative Reward: {cumulative_reward}")
            
    print("Training Complete. Displaying final Q-value policies...")
    
    # Save the cumulative reward curve
    plt.figure(figsize=(10, 5))
    # Rolling average plot
    rolling_rewards = np.convolve(rewards_history, np.ones(20)/20, mode='valid')
    plt.plot(rolling_rewards, label='Rolling Cumulative Reward (window=20)')
    plt.title('Q-Learning Cumulative Reward Convergence Curve')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.legend()
    plt.grid(True)
    plt.savefig('reward_curve.png')
    print("Saved reward curve plot to reward_curve.png")
    
if __name__ == "__main__":
    run_q_learning()
