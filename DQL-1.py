from DQN import DQN
import matplotlib.pyplot as plt
import numpy as np
import random
from Replay import ReplayBuffer
import torch
import torch.optim as optim
import torch.nn.functional as func
from typing import Dict


class DQNAgent:
    """
    DQN Agent that interacts with and learns from the environment.
    """

    def __init__(self, env, buffer_capacity: int = 10000, batch_size: int = 64, learning_rate: float = 0.001,
                 gamma: float = 0.99, epsilon: float = 1.0, epsilon_min: float = 0.01, epsilon_decay: float = 0.995,
                 target_update_interval: int = 10):
        """
        Initialize the DQN agent.

        :param env: The environment to interact with.
        :param buffer_capacity: The maximum size of the replay buffer.
        :param batch_size: The batch size for sampling from the replay buffer.
        :param learning_rate: The learning rate for the optimizer.
        :param gamma: The discount factor for future rewards.
        :param epsilon: The initial exploration rate.
        :param epsilon_min: The minimum exploration rate.
        :param epsilon_decay: The decay rate for the exploration rate.
        :param target_update_interval: The number of episodes between target network updates.
        """
        self.env = env
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_interval = target_update_interval
        self.n_actions = env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.policy_network = DQN(env.observation_space.shape[0], self.n_actions).to(self.device)
        self.target_network = DQN(env.observation_space.shape[0], self.n_actions).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

    def choose_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """
        Select an action for the given state using an epsilon-greedy policy.

        :param state: The current state of the environment.
        :param evaluate: If True, select action greedily without exploration.
        :return: The selected action.
        """
        if not evaluate and random.random() < self.epsilon:
            return self.env.action_space.sample()

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.policy_network(state_tensor).max(1)[1].item()

    def optimize_model(self):
        """
        Update the policy network by sampling a batch of experiences from the replay buffer
        and performing a gradient descent step.
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample a batch of experiences from the replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(self.batch_size)
        state_batch = state_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        done_batch = done_batch.to(self.device)

        # Q-values for current states
        q_values = self.policy_network(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)

        # Q-values for next states using target network
        next_q_values = self.target_network(next_state_batch).max(1)[0]
        expected_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values

        # Compute loss and optimize
        loss = func.mse_loss(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_agent(self, num_episodes: int) -> Dict[str, list]:
        """
        Train the agent for a specified number of episodes.

        :param num_episodes: The number of episodes to train the agent.
        :return: A dictionary with episode statistics including average, maximum, and minimum rewards.
        """
        episode_rewards = []
        statistics = {"episode": [], "average_reward": [], "max_reward": [], "min_reward": []}

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, terminated, truncation, _ = self.env.step(action)
                done = terminated or truncation

                # Store the experience in the replay buffer
                self.replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                # Update the policy network
                self.optimize_model()

            # Decay epsilon after each episode
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            episode_rewards.append(total_reward)
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                max_reward = np.max(episode_rewards[-10:])
                min_reward = np.min(episode_rewards[-10:])
                statistics["episode"].append(episode)
                statistics["average_reward"].append(avg_reward)
                statistics["max_reward"].append(max_reward)
                statistics["min_reward"].append(min_reward)

                print(
                    f"Ep: {episode:>5d}, avg: {avg_reward:>4.1f}, max: {max_reward:>4.1f}, "
                    f"min: {min_reward:>4.1f}, epsilon: {self.epsilon:>1.2f}"
                )

            # Update the target network every `target_update_interval` episodes
            if episode % self.target_update_interval == 0:
                self.target_network.load_state_dict(self.policy_network.state_dict())

        return statistics

    def plot_training_progress(self, statistics: Dict[str, list], num_episodes: int):
        """
        Display the training progress using matplotlib.

        :param statistics: A dictionary with episode statistics.
        :param num_episodes: The total number of episodes.
        """
        plt.plot(statistics['episode'], statistics['average_reward'], label="average rewards")
        plt.plot(statistics['episode'], statistics['max_reward'], label="max rewards")
        plt.plot(statistics['episode'], statistics['min_reward'], label="min rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend(loc=4)
        plt.title("Rewards per Episode (DQN Training)")
        plt.grid(True)
        plt.show()
