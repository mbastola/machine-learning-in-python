import gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict

# Use a consistent style for plots
plt.style.use('seaborn-v0_8-whitegrid')

# Initialize the environment
env = gym.make('FrozenLake-v1', is_slippery=True, render_mode='rgb_array')
unwrapped_env = env.unwrapped # For accessing the model P

n_states = unwrapped_env.observation_space.n
n_actions = unwrapped_env.action_space.n

action_names = {0: 'Left', 1: 'Down', 2: 'Right', 3: 'Up'}
action_symbols = {0: '←', 1: '↓', 2: '→', 3: '↑'}

print(f"Number of States: {n_states}")
print(f"Number of Actions: {n_actions}")

def value_iteration(env, gamma=0.99, theta=1e-8):
    """Performs Value Iteration to find the optimal value function."""
    V = np.zeros(env.observation_space.n)
    while True:
        delta = 0
        for s in range(env.observation_space.n):
            v = V[s]
            q_sa = np.zeros(env.action_space.n)
            for a in range(env.action_space.n):
                for prob, next_s, reward, _ in env.P[s][a]:
                    q_sa[a] += prob * (reward + gamma * V[next_s])
            V[s] = np.max(q_sa)
            delta = max(delta, np.abs(v - V[s]))
        if delta < theta:
            break
    policy = np.zeros(env.observation_space.n, dtype=int)
    for s in range(env.observation_space.n):
        q_sa = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for prob, next_s, reward, _ in env.P[s][a]:
                q_sa[a] += prob * (reward + gamma * V[next_s])
        policy[s] = np.argmax(q_sa)
    return V, policy

optimal_V_vi, optimal_policy_vi = value_iteration(unwrapped_env)

def plot_results(V, policy, title):
    """Helper function to plot the value function and policy."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=16)
    
    V_reshaped = V.reshape(4, 4)
    sns.heatmap(V_reshaped, annot=True, fmt=".3f", cmap="viridis", cbar=False, ax=ax[0], linewidths=.5)
    ax[0].set_title("Value Function (V)")
    ax[0].set_xlabel("Columns")
    ax[0].set_ylabel("Rows")
    
    policy_reshaped = policy.reshape(4, 4)
    ax[1].imshow(V_reshaped, cmap="viridis")
    ax[1].set_title("Policy (π)")
    ax[1].set_xlabel("Columns")
    ax[1].set_ylabel("Rows")
    
    for r in range(4):
        for c in range(4):
            action = policy_reshaped[r, c]
            arrow = action_symbols[action]
            ax[1].text(c, r, arrow, ha='center', va='center', color='red', fontsize=20)
            
    desc = unwrapped_env.desc.astype(str)
    for r in range(4):
        for c in range(4):
            if desc[r, c] in 'HG':
                ax[0].text(c + 0.5, r + 0.5, desc[r, c], ha='center', va='center', color='white', fontsize=20)
                ax[1].text(c + 0.5, r + 0.5, desc[r, c], ha='center', va='center', color='white', fontsize=20)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

plot_results(optimal_V_vi, optimal_policy_vi, "Baseline: Results from Value Iteration")

def on_policy_first_visit_mc_control(env, n_episodes, gamma, epsilon):
    """
    On-policy first-visit Monte Carlo control for ε-soft policies.
    """
    # Initialize Q(s, a) and the return tracking dictionaries
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    
    # The policy is derived from Q, but we can think of it as ε-greedy
    def get_policy(state):
        probs = np.ones(env.action_space.n) * epsilon / env.action_space.n
        best_action = np.argmax(Q[state])
        probs[best_action] += 1.0 - epsilon
        return probs

    rewards_per_episode = []

    for i in range(n_episodes):
        if (i + 1) % 50000 == 0:
            print(f"Episode {i + 1}/{n_episodes}")
            
        # 1. Generate an episode
        episode_history = []
        state, info = env.reset()
        terminated = False
        truncated = False
        
        while not terminated and not truncated:
            policy_probs = get_policy(state)
            action = np.random.choice(np.arange(env.action_space.n), p=policy_probs)
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_history.append((state, action, reward))
            state = next_state
        
        # Track success for learning curve
        rewards_per_episode.append(reward)

        # 2. Update Q-values (Policy Evaluation)
        G = 0
        visited_sa_pairs = set()
        # Iterate backwards through the episode
        for t in range(len(episode_history) - 1, -1, -1):
            state, action, reward = episode_history[t]
            G = gamma * G + reward
            
            # First-visit MC: only update on the first time we see (s, a)
            if (state, action) not in visited_sa_pairs:
                visited_sa_pairs.add((state, action))
                
                returns_sum[(state, action)] += G
                returns_count[(state, action)] += 1.0
                Q[state][action] = returns_sum[(state, action)] / returns_count[(state, action)]
                # Policy is implicitly improved because get_policy() uses the updated Q

    # Derive final deterministic policy and V function from Q
    final_policy = np.zeros(env.observation_space.n, dtype=int)
    V = np.zeros(env.observation_space.n)
    for s in range(env.observation_space.n):
        final_policy[s] = np.argmax(Q[s])
        V[s] = np.max(Q[s])
    
    return V, final_policy, rewards_per_episode

# Monte Carlo parameters
# MC methods often require many more episodes than TD methods due to high variance.
n_episodes_mc = 500000 
gamma_mc = 0.99
epsilon_mc = 0.1 # For the ε-soft policy

print("--- Running On-Policy First-Visit MC Control ---")
V_mc, policy_mc, rewards_mc = on_policy_first_visit_mc_control(
    env, n_episodes_mc, gamma_mc, epsilon_mc
)

plot_results(V_mc, policy_mc, "Monte Carlo Control Results")

def plot_learning_curve(rewards, window_size=1000):
    """Plots the success rate over time."""
    # Calculate success rate over a sliding window
    success_rate = pd.Series(rewards).rolling(window_size, min_periods=1).mean()
    
    plt.figure(figsize=(12, 6))
    plt.plot(success_rate, label='Monte Carlo')
    plt.title('Learning Curve: Success Rate Over Time')
    plt.xlabel('Episode')
    plt.ylabel(f'Success Rate (Avg over last {window_size} episodes)')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_learning_curve(rewards_mc)

print("--- Final Policy Comparison ---")
policy_match = np.array_equal(optimal_policy_vi, policy_mc)
print(f"Value Iteration vs. Monte Carlo policies identical: {policy_match}")

if not policy_match:
    diffs = np.sum(optimal_policy_vi != policy_mc)
    print(f"Number of differing actions in policies: {diffs} out of {n_states}")

env.close()