import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

# Constants
LR = 0.1  # Learning rate
Discount = 0.95  # Discount factor for future rewards
EP = 30000  # Number of episodes
check = 1000  # Checkpoint interval for rendering

# Initialize the environment
env = gym.make("CartPole-v1")
obs_space_high = np.array([env.observation_space.high[0], 0.5, env.observation_space.high[2], 0.5])
obs_space_low = np.array([env.observation_space.low[0], -0.5, env.observation_space.low[2], -0.5])

discrete_observation_size = [20, 20, 20, 20]
discrete_observation_win_size = (obs_space_high - obs_space_low) / discrete_observation_size

# Exploration settings
epsilon = 1  # Initial epsilon for exploration
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EP // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# Initialize Q-table with random values
Q_table = np.random.uniform(
    low=-2, high=0, size=(discrete_observation_size + [env.action_space.n])
)
rewards_per_ep = []
stat_rewards = {"ep": [], "avg": [], "max": [], "min": []}

def discretize_state(state):
    """
    Discretize state variables
    :param state: the state
    :return: discrete_state
    """
    state = np.clip(state, obs_space_low, obs_space_high)
    discrete_state = (state - obs_space_low) / discrete_observation_win_size
    return tuple(np.clip(discrete_state.astype(int), 0, np.array(discrete_observation_size) - 1))

for ep in range(EP):
    current_rewards_per_ep = 0
    # Reset the environment and get the initial state
    render = ep % check == 0
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    initial_state, _ = env.reset()
    discrete_state = discretize_state(initial_state)

    done = False

    if render:
        print(f"Episode: {ep}")

    while not done:
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(Q_table[discrete_state])
        else:
            # Get random action
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, termination, truncation, _ = env.step(action)
        current_rewards_per_ep += reward
        done = termination or truncation
        new_discrete_state = discretize_state(new_state)

        if render:
            env.render()

        if not done:
            max_next_Q = np.max(Q_table[new_discrete_state])  # Max Q
            Q = Q_table[discrete_state + (action,)]  # Current Q
            # Q learning eq
            new_Q = (1 - LR) * Q + LR * (reward + Discount * max_next_Q)
            # Update Q table
            Q_table[discrete_state + (action,)] = new_Q
        elif termination:
            Q_table[discrete_state + (action,)] = 0  # Update Q value for terminal state

        discrete_state = new_discrete_state

    # Track rewards
    rewards_per_ep.append(current_rewards_per_ep)

    if ep % 100 == 0:
        average_reward = np.mean(rewards_per_ep[-100:])
        stat_rewards["ep"].append(ep)
        stat_rewards["avg"].append(average_reward)
        stat_rewards["max"].append(np.max(rewards_per_ep[-100:]))
        stat_rewards["min"].append(np.min(rewards_per_ep[-100:]))

    # Decaying epsilon
    if END_EPSILON_DECAYING >= ep >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

env.close()

# Set Matplotlib backend to 'Agg' for headless environments
plt.switch_backend('Agg')

plt.plot(stat_rewards['ep'], stat_rewards['avg'], label="average rewards")
plt.plot(stat_rewards['ep'], stat_rewards['max'], label="max rewards")
plt.plot(stat_rewards['ep'], stat_rewards['min'], label="min rewards")
plt.legend(loc=4)
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('Rewards per Episode (CartPole)')
plt.savefig('rewards_plot.png')
