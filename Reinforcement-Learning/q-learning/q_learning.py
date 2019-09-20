import gym
import numpy as np

env = gym.make('FrozenLake-v1', is_slippery=True, render_mode='rgb_array')

print("Observation Space (States):", env.observation_space)
print("Action Space (Actions):", env.action_space)

state, info = env.reset()
print("\nInitial State:", state)

# The states are the positions on our 4x4 grid. There are 16 states, indexed from 0 to 15.
n_states = env.observation_space.n
print("Number of States:", n_states)

#### Actions ($A$)

# The actions are the movements the agent can make.
# * 0: Left
# * 1: Down
# * 2: Right
# * 3: Up
n_actions = env.action_space.n
print("Number of Actions:", n_actions)

#### Transition Probabilities ($P$)

# The transition probabilities define the dynamics of the environment. The `env.P` attribute is a dictionary
# that holds this information. `P[state][action]` gives us a list of tuples `(prob, next_state, reward, terminated)`.

# Let's examine the transitions from state 0 (the start state) for action 1 (Down).
# The slippery nature of the ice means an intended action might not have the intended outcome.
print("Transitions from State 0 (Action: Down):")
print(env.P[0][1])

#### Rewards ($R$)

# The rewards are given for reaching a certain state. In our environment:
# * Reaching the goal 'G' (state 15): reward of 1.0
# * Falling into a hole 'H': reward of 0.0
# * All other steps: reward of 0.0

# Let's verify the reward for reaching the goal state (15).
# For example, from state 14, taking action 2 (Right) should lead to state 15.
print("Transitions from State 14 (Action: Right):")
print(env.P[14][2])

### 4. Solving the MDP with Value Iteration
# The Bellman optimality equation for the value function is:
# $$V_{k+1}(s) = \max_{a \in A} \sum_{s', r} P(s'|s,a) [R(s,a,s') + \gamma V_k(s')]$$

def value_iteration(env, gamma=0.99, theta=1e-8):
    """
    Performs Value Iteration to find the optimal value function.
    """
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

# Run the value iteration algorithm
optimal_V, optimal_policy_vi = value_iteration(env)

print("Optimal Value Function (V*) from Value Iteration:\n", np.round(optimal_V.reshape(4, 4), 3))
print("\nOptimal Policy (from Value Iteration):\n", optimal_policy_vi.reshape(4, 4))

# The Bellman expectation equation (policy evaluation): 
# $$V_{k+1}^{\pi}(s) = \sum_{a} \pi(a|s) \sum_{s', r} P(s'|s,a) [R(s,a,s') + \gamma V_k^{\pi}(s')]$$

def policy_evaluation(env, policy, gamma=0.99, theta=1e-8):
    """
    Evaluates the value function for a given policy.
    """
    V = np.zeros(env.observation_space.n)
    while True:
        delta = 0
        for s in range(env.observation_space.n):
            v_old = V[s]
            v_new = 0
            a = policy[s]
            for prob, next_s, reward, _ in env.P[s][a]:
                v_new += prob * (reward + gamma * V[next_s])
            V[s] = v_new
            delta = max(delta, abs(v_old - V[s]))
        if delta < theta:
            break
    return V

def policy_iteration(env, gamma=0.99):
    """
    Performs Policy Iteration to find the optimal policy and value function.
    """
    policy = np.random.randint(env.action_space.n, size=env.observation_space.n)
    
    while True:
        # Step 1: Policy Evaluation
        V = policy_evaluation(env, policy, gamma)
        
        # Step 2: Policy Improvement
        policy_stable = True
        new_policy = np.copy(policy)
        for s in range(env.observation_space.n):
            old_action = policy[s]
            
            q_sa = np.zeros(env.action_space.n)
            for a in range(env.action_space.n):
                for prob, next_s, reward, _ in env.P[s][a]:
                    q_sa[a] += prob * (reward + gamma * V[next_s])
            
            best_action = np.argmax(q_sa)
            new_policy[s] = best_action
            
            if old_action != best_action:
                policy_stable = False
                
        policy = new_policy
        
        if policy_stable:
            break
            
    return V, policy

# Run the policy iteration
optimal_V_pi, optimal_policy_pi = policy_iteration(env)

print("\nOptimal Value Function (V*) from Policy Iteration:\n", np.round(optimal_V_pi.reshape(4, 4), 3))
print("\nOptimal Policy (from Policy Iteration):\n", optimal_policy_pi.reshape(4, 4))

# Check if the policies are the same (they should be)
print("\nPolicies from both methods are identical:", np.array_equal(optimal_policy_vi, optimal_policy_pi))


### 5. Temporal Difference Learning (Model-Free)

# Temporal Difference (TD) learning methods are model-free, meaning they don't require knowledge of the environment's transition probabilities or reward function. They learn directly from experience.

# We'll implement two common TD control algorithms: SARSA (on-policy) and Q-learning (off-policy).

def epsilon_greedy_policy(Q, state, n_actions, epsilon):
    """
    Epsilon-greedy policy for action selection.
    """
    if np.random.rand() < epsilon:
        return env.action_space.sample()  # Explore
    else:
        return np.argmax(Q[state, :])  # Exploit

def get_action(Q, state):
    """
    Get the best action from the Q-table for a given state.
    """
    return np.argmax(Q[state, :])


#### SARSA (On-Policy TD Control)

# SARSA learns the action-value function $Q(s,a)$ based on the current policy. The update rule is:
# $$Q(s,a) \leftarrow Q(s,a) + \alpha [R_{t+1} + \gamma Q(s',a') - Q(s,a)]$$
# where $a'$ is the action chosen in state $s'$ according to the *current* policy.

def sarsa(env, n_episodes, alpha, gamma, epsilon_start, epsilon_end, epsilon_decay):
    """
    SARSA algorithm for learning optimal policy.
    """
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    epsilon = epsilon_start

    for episode in range(n_episodes):
        state, info = env.reset()
        terminated = False
        truncated = False
        action = epsilon_greedy_policy(Q, state, env.action_space.n, epsilon)

        while not terminated and not truncated:
            next_state, reward, terminated, truncated, info = env.step(action)
            next_action = epsilon_greedy_policy(Q, next_state, env.action_space.n, epsilon)

            # SARSA update
            old_value = Q[state, action]
            next_q = Q[next_state, next_action]
            
            new_value = old_value + alpha * (reward + gamma * next_q - old_value)
            Q[state, action] = new_value

            state = next_state
            action = next_action
        
        epsilon = max(epsilon_end, epsilon * epsilon_decay) # Decay epsilon

    # Derive optimal policy and value function from Q
    policy = np.argmax(Q, axis=1)
    V = np.max(Q, axis=1)
    
    return Q, V, policy

# SARSA parameters
n_episodes_sarsa = 20000
alpha_sarsa = 0.1 # Learning rate
gamma_sarsa = 0.99 # Discount factor
epsilon_start_sarsa = 1.0
epsilon_end_sarsa = 0.01
epsilon_decay_sarsa = 0.9995

print("\n--- Running SARSA ---")
optimal_Q_sarsa, optimal_V_sarsa, optimal_policy_sarsa = sarsa(env, n_episodes_sarsa, alpha_sarsa, gamma_sarsa, 
                                                              epsilon_start_sarsa, epsilon_end_sarsa, epsilon_decay_sarsa)

print("Optimal Value Function (V*) from SARSA:\n", np.round(optimal_V_sarsa.reshape(4, 4), 3))
print("\nOptimal Policy (from SARSA):\n", optimal_policy_sarsa.reshape(4, 4))


#### Q-Learning (Off-Policy TD Control)

# Q-learning also learns the action-value function $Q(s,a)$, but it's an off-policy algorithm. The update rule is:
# $$Q(s,a) \leftarrow Q(s,a) + \alpha [R_{t+1} + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$
# Here, $a'$ is the action that maximizes $Q(s',a')$, regardless of the action actually taken to transition to $s'$.

def q_learning(env, n_episodes, alpha, gamma, epsilon_start, epsilon_end, epsilon_decay):
    """
    Q-learning algorithm for learning optimal policy.
    """
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    epsilon = epsilon_start

    for episode in range(n_episodes):
        state, info = env.reset()
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = epsilon_greedy_policy(Q, state, env.action_space.n, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)

            # Q-learning update
            old_value = Q[state, action]
            max_next_q = np.max(Q[next_state, :]) # Max Q for next state
            
            new_value = old_value + alpha * (reward + gamma * max_next_q - old_value)
            Q[state, action] = new_value

            state = next_state
        
        epsilon = max(epsilon_end, epsilon * epsilon_decay) # Decay epsilon

    # Derive optimal policy and value function from Q
    policy = np.argmax(Q, axis=1)
    V = np.max(Q, axis=1)
    
    return Q, V, policy

# Q-learning parameters
n_episodes_q_learning = 20000
alpha_q_learning = 0.1 # Learning rate
gamma_q_learning = 0.99 # Discount factor
epsilon_start_q_learning = 1.0
epsilon_end_q_learning = 0.01
epsilon_decay_q_learning = 0.9995

print("\n--- Running Q-Learning ---")
optimal_Q_q_learning, optimal_V_q_learning, optimal_policy_q_learning = q_learning(env, n_episodes_q_learning, alpha_q_learning, gamma_q_learning,
                                                                                  epsilon_start_q_learning, epsilon_end_q_learning, epsilon_decay_q_learning)

print("Optimal Value Function (V*) from Q-Learning:\n", np.round(optimal_V_q_learning.reshape(4, 4), 3))
print("\nOptimal Policy (from Q-Learning):\n", optimal_policy_q_learning.reshape(4, 4))


# Final comparison of policies
print("\nPolicies from Value Iteration and SARSA are identical:", np.array_equal(optimal_policy_vi, optimal_policy_sarsa))
print("Policies from Value Iteration and Q-Learning are identical:", np.array_equal(optimal_policy_vi, optimal_policy_q_learning))
print("Policies from SARSA and Q-Learning are identical:", np.array_equal(optimal_policy_sarsa, optimal_policy_q_learning))

env.close()