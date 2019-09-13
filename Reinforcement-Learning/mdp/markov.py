
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


env.close()
