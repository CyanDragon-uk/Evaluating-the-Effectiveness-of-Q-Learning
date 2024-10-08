import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

# Constants
LR = 0.1  # 0-1
Discount = 0.95  # measure of importance for future actions
EP = 30000
check = 1000  # checkpoint for the task

# env = gym.make("MountainCar-v0", render_mode="human")
# state = env.reset()
# env.reset()
env = gym.make("MountainCar-v0")
discrete_observation_size = [20] * len(env.observation_space.high)
discrete_observation_win_size = (
    env.observation_space.high - env.observation_space.low
) / discrete_observation_size
# print(discrete_observation_win_size)

# print(env.observation_space.high)
# print(env.observation_space.low)
# print(env.action_space.n)

# Exploration
epsilon = 1  # higher for random action
Decay_Start = 1
End_Decay = EP // 2
epsilon_value = epsilon / (End_Decay - Decay_Start)

Q_table = np.random.uniform(
    low=-2, high=0, size=(discrete_observation_size + [env.action_space.n])
)
rewards_per_ep = []
stat_rewards = {"ep": [], "avg": [], "max": [], "min": []}


def discretize_state(state):
    """

    :param state:
    :return:
    """
    discrete_state = (state - env.observation_space.low) / discrete_observation_win_size
    return tuple(discrete_state.astype(int))


for ep in range(EP):
    # initialise current reward
    current_rewards_per_ep = 0
    # Reset the environment and get the initial state
    render = ep % check == 0
    env = gym.make("MountainCar-v0", render_mode="human" if render else None)
    initial_state, _ = env.reset()
    discrete_state = discretize_state(initial_state)
    # print(discrete_state)
    # print(Q_table[discrete_state])
    # print(np.argmax(Q_table[discrete_state]))
    done = False

    # FOR TESTING
    if render:
        print(f"Episode: {ep}")
    while not done:
        # action = 1
        # env.step(action)
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(Q_table[discrete_state])
        else:
            # Get random action
            action = np.random.randint(0, env.action_space.n)
        new_state, reward, termination, truncation, _ = env.step(action)
        # print(reward, new_state)
        current_rewards_per_ep += reward
        done = termination or truncation
        new_discrete_state = discretize_state(new_state)
        if render:
            env.render()
        if (
            not done
        ):  # https://wikimedia.org/api/rest_v1/media/math/render/svg/a3a4d2ac903b1be02cc81e60de2e9f91d7025fec
            max_next_Q = np.max(Q_table[new_discrete_state])  # Max Q
            Q = Q_table[discrete_state + (action,)]  # Current Q
            # Q learning eq
            new_Q = (1 - LR) * Q + LR * (reward + Discount * max_next_Q)
            # Update Q table
            Q_table[discrete_state + (action,)] = new_Q

        elif new_state[0] >= env.unwrapped.goal_position:
            Q_table[discrete_state + (action,)] = 0  # Update Q value
        discrete_state = new_discrete_state

    # Decaying epsilon
    if End_Decay >= ep >= Decay_Start:
        epsilon -= epsilon_value
    rewards_per_ep.append(current_rewards_per_ep)
    if not ep % check:
        avg_reward = sum(rewards_per_ep[-check:]) / len(rewards_per_ep[-check:])
        stat_rewards["ep"].append(ep)
        stat_rewards["avg"].append(avg_reward)
        stat_rewards["max"].append(max(rewards_per_ep[-check:]))
        stat_rewards["min"].append(min(rewards_per_ep[-check:]))
        print(
            f"Ep: {ep:>5d}, avg: {avg_reward:>4.1f}, max: {max(rewards_per_ep[-check:]):>4.1f}, "
            f"min: {min(rewards_per_ep[-check:]):>4.1f}, epsilon: {epsilon:>1.2f}"
        )

env.close()

plt.plot(stat_rewards['ep'], stat_rewards['avg'], label="average rewards")
plt.plot(stat_rewards['ep'], stat_rewards['max'], label="max rewards")
plt.plot(stat_rewards['ep'], stat_rewards['min'], label="min rewards")
plt.legend(loc=4)
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('Rewards per Episode (MountainCar)')
plt.show()
