import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# Use the 'Agg' backend for matplotlib
matplotlib.use('Agg')


def discretize_state(env, observation, bins):
    """
    Discretize continuous state variables into discrete bins.

    :param env: The environment object
    :param observation: The current observation from the environment
    :param bins: The number of bins for discretisation
    :return: Discretized state as a tuple
    """
    upper_bounds = [env.observation_space.high[0], 1.0, env.observation_space.high[2], 1.0, 1.0, 1.0, 1.0, 1.0]
    lower_bounds = [env.observation_space.low[0], -1.0, env.observation_space.low[2], -1.0, -1.0, -1.0, -1.0, -1.0]
    ratios = [(observation[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in
              range(len(observation))]
    new_obs = [int(round((bins[i] - 1) * ratios[i])) for i in range(len(observation))]
    new_obs = [min(bins[i] - 1, max(0, new_obs[i])) for i in range(len(observation))]
    return tuple(new_obs)

def initialize_q_table(env, bins):
    """
    Initialize the Q-table for all state-action pairs.

    :param env: The environment object
    :param bins: The number of bins for discretization
    :return: Initialized Q-table
    """
    dimension_sizes = bins + [env.action_space.n]
    q_table = np.zeros(dimension_sizes, dtype=float)
    return q_table


def choose_action(q_table, state, epsilon):
    """
    Choose an action using an epsilon-greedy policy.

    :param q_table: The Q-table
    :param state: The current state
    :param epsilon: The exploration rate
    :return: Chosen action
    """
    if np.random.random() < epsilon:
        return np.random.randint(q_table.shape[-1])  # Explore
    else:
        return np.argmax(q_table[state])  # Exploit


def update_q_value(q_table, state, action, reward, next_state, alpha, gamma):
    """
    Update the Q-value for a given state and action.

    :param q_table: The Q-table
    :param state: The current state
    :param action: The action taken
    :param reward: The reward received
    :param next_state: The next state
    :param alpha: The learning rate
    :param gamma: The discount factor
    :return: None
    """
    current_q = q_table[state][action]
    max_future_q = np.max(q_table[next_state])
    new_q = (1 - alpha) * current_q + alpha * (reward + gamma * max_future_q)
    q_table[state][action] = new_q


def run_q_learning(env, bins, episodes, alpha, gamma, epsilon, min_epsilon, epsilon_decay, checkpoint):
    """
    Run the Q-learning algorithm over a number of episodes.

    :param env: The environment object
    :param bins: The number of bins for discretization
    :param episodes: The number of episodes to run
    :param alpha: The learning rate
    :param gamma: The discount factor
    :param epsilon: The initial exploration rate
    :param min_epsilon: The minimum exploration rate
    :param epsilon_decay: The decay rate for epsilon
    :param checkpoint: Interval for logging statistics
    :return: Trained Q-table and statistics
    """
    q_table = initialize_q_table(env, bins)
    rewards_per_ep = []
    stat_rewards = {"ep": [], "avg": [], "max": [], "min": []}

    for episode in range(episodes):
        current_rewards_per_ep = 0
        render = episode % checkpoint == 0
        env = gym.make("LunarLander-v2", render_mode="human" if render else None)
        state = discretize_state(env, env.reset(seed=42)[0], bins)
        done = False

        while not done:
            action = choose_action(q_table, state, epsilon)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            next_state = discretize_state(env, next_observation, bins)

            # Reward shaping: add small reward for each step and adjust for smooth landing
            if terminated:
                if next_observation[6] < 0.05 and next_observation[7] < 0.05:
                    reward += 100  # Bonus for soft landing
                else:
                    reward -= 100  # Penalty for crash
            elif truncated:
                reward -= 50  # Penalty for timeout
            else:
                reward -= np.abs(next_observation[6]) + np.abs(next_observation[7])  # Penalty for instability

            update_q_value(q_table, state, action, reward, next_state, alpha, gamma)
            state = next_state
            current_rewards_per_ep += reward
            done = terminated or truncated

        rewards_per_ep.append(current_rewards_per_ep)
        if not episode % checkpoint:
            avg_reward = sum(rewards_per_ep[-checkpoint:]) / len(rewards_per_ep[-checkpoint:])
            stat_rewards["ep"].append(episode)
            stat_rewards["avg"].append(avg_reward)
            stat_rewards["max"].append(max(rewards_per_ep[-checkpoint:]))
            stat_rewards["min"].append(min(rewards_per_ep[-checkpoint:]))
            print(
                f"Ep: {episode:>5d}, avg: {avg_reward:>4.1f}, max: {max(rewards_per_ep[-checkpoint:]):>4.1f}, "
                f"min: {min(rewards_per_ep[-checkpoint:]):>4.1f}, epsilon: {epsilon:>1.2f}"
            )

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    return q_table, stat_rewards


# Parameters for Q-learning
bins = [30, 30, 30, 30, 5, 5, 5, 5]
episodes = 30000
alpha = 0.1
gamma = 0.99
epsilon = 1.0
min_epsilon = 0.01
epsilon_decay = episodes / 2
checkpoint = 1000

# Running the Q-learning on LunarLander environment
env = gym.make("LunarLander-v2")
q_table_trained, stat_rewards = run_q_learning(env, bins, episodes, alpha, gamma, epsilon, min_epsilon, epsilon_decay,
                                               checkpoint)
env.close()

# Save the Q-table for further use or analysis
np.save("lunarlander_q_table.npy", q_table_trained)

# Plotting the statistics
plt.plot(stat_rewards['ep'], stat_rewards['avg'], label="average rewards")
plt.plot(stat_rewards['ep'], stat_rewards['max'], label="max rewards")
plt.plot(stat_rewards['ep'], stat_rewards['min'], label="min rewards")
plt.legend(loc=4)
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('Rewards per Episode (LunarLander)')
plt.savefig('rewards_plot.png')  # Save the plot as an image file
plt.close()  # Close the plot to release resources

