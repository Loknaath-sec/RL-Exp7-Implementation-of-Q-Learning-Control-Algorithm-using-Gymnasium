# EXP 7 - Implementation of Q-Learning Control Algorithm Using Gymnasium

## Aim

To implement the **Q-Learning control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an optimal action-value function that enables the agent to select suitable actions for reaching the goal state while avoiding holes.

---

## Problem Statement

The objective is to implement a model-free **Q-Learning control algorithm** for the `FrozenLake-v1` environment provided by Gymnasium.

The agent must learn the best action to take in each state by interacting with the environment. The goal is to reach the destination while avoiding holes.

Since the environment is stochastic when `is_slippery=True`, the agent must learn a policy that performs well despite uncertain state transitions.

---

## Software Requirements

* Python
* Jupyter Notebook / Google Colab / VS Code

### Installation

Install the required Python packages using:

```bash
pip install gymnasium numpy matplotlib
```

---

## Environment Description

The experiment uses the **FrozenLake-v1** environment from Gymnasium.

FrozenLake is a grid-world reinforcement learning environment in which an agent must navigate across a frozen lake and reach the goal without falling into holes.

### Grid Symbols

| Symbol | Description    |
| ------ | -------------- |
| `S`    | Starting point |
| `F`    | Frozen surface |
| `H`    | Hole           |
| `G`    | Goal           |

### Environment Configuration

```python
env = gym.make(
    "FrozenLake-v1",
    map_name="4x4",
    is_slippery=True
)
```

The 4 × 4 environment contains:

* **16 states**
* **4 actions**

### Actions

| Action | Meaning |
| -----: | ------- |
|    `0` | Left    |
|    `1` | Down    |
|    `2` | Right   |
|    `3` | Up      |

### Rewards

| Event               | Reward |
| ------------------- | -----: |
| Normal movement     |    `0` |
| Falling into a hole |    `0` |
| Reaching the goal   |    `1` |

When `is_slippery=True`, the agent may not always move in the intended direction, making the environment stochastic.

---

## Theory

Q-Learning is a **model-free reinforcement learning algorithm** that learns the optimal action-value function without requiring a model of the environment.

The action-value function `Q(s, a)` represents the expected return obtained by taking action `a` in state `s` and subsequently following the best possible policy.

### Q-Learning Update Rule

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) +
\alpha \left[
R_{t+1} +
\gamma \max_a Q(S_{t+1}, a) -
Q(S_t, A_t)
\right]
$$

Where:

| Symbol          | Meaning                           |
| --------------- | --------------------------------- |
| `Sₜ`            | Current state                     |
| `Aₜ`            | Current action                    |
| `Rₜ₊₁`          | Reward received                   |
| `Sₜ₊₁`          | Next state                        |
| `α`             | Learning rate                     |
| `γ`             | Discount factor                   |
| `Q(s,a)`        | Action-value function             |
| `max Q(Sₜ₊₁,a)` | Maximum Q-value of the next state |

---

## Epsilon-Greedy Action Selection

During training, the agent uses an **epsilon-greedy strategy** to balance exploration and exploitation.

With probability `ε`, the agent chooses a random action.

With probability `1 - ε`, the agent chooses the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_a Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

Initially, epsilon is high so that the agent explores the environment. As training progresses, epsilon is reduced so that the agent increasingly exploits the learned Q-values.

---

## Algorithm

1. Initialize the FrozenLake environment.
2. Determine the number of states and actions.
3. Initialize the Q-table with zeros.
4. Set the learning rate `α`.
5. Set the discount factor `γ`.
6. Initialize epsilon `ε`.
7. Repeat for the specified number of episodes:
   * Reset the environment.
   * Obtain the initial state.
   * Select an action using epsilon-greedy action selection.
   * Execute the action.
   * Observe the next state and reward.
   * Update the Q-value using the Q-Learning equation.
   * Move to the next state.
   * Continue until the episode terminates.
8. Decrease epsilon after each episode.
9. Calculate the estimated state-value function.
10. Extract the learned policy using the maximum Q-value for each state.
11. Calculate the average reward over the final 1000 episodes.
12. Plot the training performance.

---

## Python Program
```
Name: Loknaath P
Reg.No: 212223240080
```

```py
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
```

```py
# -------------------------------------------------
# Create FrozenLake Environment
# -------------------------------------------------

env = gym.make("FrozenLake-v1", is_slippery=True)

n_states = env.observation_space.n
n_actions = env.action_space.n
```

```py
# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------

learning_rate = 0.8
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
num_episodes = 20000
```

```py
# -------------------------------------------------
# Initialize Q-table
# -------------------------------------------------

Q = np.zeros((n_states, n_actions))


# -------------------------------------------------
# Epsilon-Greedy Action Selection
# -------------------------------------------------

def choose_action(state, epsilon):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    return int(np.argmax(Q[state]))
```

```py
# -------------------------------------------------
# Q-Learning Training
# -------------------------------------------------

episode_rewards = []

for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = choose_action(state, epsilon)

        next_state, reward, terminated, truncated, _ = env.step(action)

        # Q-learning update:
        # Q(s,a) <- Q(s,a) + alpha * [r + gamma*max Q(s',a') - Q(s,a)]
        if terminated:
            target = reward
        else:
            target = reward + gamma * np.max(Q[next_state])

        Q[state, action] += learning_rate * (target - Q[state, action])

        state = next_state
        total_reward += reward

    episode_rewards.append(total_reward)

    # Gradually reduce exploration.
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

# Derive the state-value function and greedy policy from the learned Q-table.
state_values = np.max(Q, axis=1)
learned_policy = np.argmax(Q, axis=1)
```

```py
# -------------------------------------------------
# Display Functions
# -------------------------------------------------

def print_value_function(values):
    print("\nEstimated State-Value Function:")
    print(np.round(values.reshape(4, 4), 3))


def print_policy(policy):
    action_symbols = {
        0: "L",
        1: "D",
        2: "R",
        3: "U"
    }

    policy_grid = np.array(
        [action_symbols[action] for action in policy]
    ).reshape(4, 4)

    print("\nLearned Policy:")
    print(policy_grid)
```

```py
# -------------------------------------------------
# Output
# -------------------------------------------------

print("\nFinal Q-table:")
print(np.round(Q, 3))

print_value_function(state_values)
print_policy(learned_policy)


average_reward = np.mean(episode_rewards[-1000:])
print("\nAverage reward over last 1000 episodes:", average_reward)
```

```py
# -------------------------------------------------
# Plot Learning Curve
# -------------------------------------------------
print("Name: LOKNAATH P")
print("Reg No: 212223240080")
window = 500

moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(8, 5))
plt.plot(moving_average)
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Q-Learning Curve - FrozenLake")
plt.grid(True)
plt.show()

env.close()
```

---

## Output

The output will vary slightly between executions because `FrozenLake-v1` with `is_slippery=True` is stochastic.

### Number of States and Actions
<img width="221" height="60" alt="image" src="https://github.com/user-attachments/assets/cdd348e7-0ee2-42e9-a3a4-dd854178c216" />

### Final Q-Table
<img width="311" height="392" alt="image" src="https://github.com/user-attachments/assets/dd180566-8244-41c1-b3f5-543e7e42ae04" />

### Estimated State-Value Function
<img width="397" height="137" alt="image" src="https://github.com/user-attachments/assets/b6ac42e2-7ff7-4b1e-adb8-1e137a9f5a2b" />

### Learned Policy
<img width="237" height="126" alt="image" src="https://github.com/user-attachments/assets/20c503ac-a747-4094-b4b9-614452d0d9c9" />

### Average Reward
<img width="478" height="42" alt="image" src="https://github.com/user-attachments/assets/229535f9-92cd-4349-aa13-8aaa9b82c685" />


## Result

The **Q-Learning control algorithm was successfully implemented** using the Gymnasium `FrozenLake-v1` environment.

The agent learned an action-value function through repeated interaction with the environment. The learned Q-table was used to determine the best action for each state.

The trained agent learned a policy for navigating toward the goal while avoiding holes.

---

## Inference

The experiment demonstrates that Q-Learning can learn an effective control policy without prior knowledge of the environment's transition model.

Initially, the agent performs more exploration because epsilon is high. As training progresses, epsilon decreases and the agent increasingly exploits the learned Q-values.

The final Q-table contains the estimated values of all actions for each state. The action with the maximum Q-value is selected as the learned policy.

Since `FrozenLake-v1` is stochastic when `is_slippery=True`, the agent may not reach the goal in every episode. Nevertheless, after sufficient training, Q-Learning learns a significantly better policy than random action selection.

---
