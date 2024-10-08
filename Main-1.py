import gymnasium as gym
from DQL import DQNAgent


def main():
    """
    Main function to train and evaluate a DQN agent on the Lunar Lander environment.
    """
    # Initialize the Lunar Lander environment
    env = gym.make("LunarLander-v2")

    # Customize environment parameters if applicable
    if hasattr(env.unwrapped, 'gravity'):
        env.unwrapped.gravity = -100.0
    if hasattr(env.unwrapped, 'enable_wind'):
        env.unwrapped.enable_wind = False
    if hasattr(env.unwrapped, 'wind_power'):
        env.unwrapped.wind_power = 15.0
    if hasattr(env.unwrapped, 'turbulence_power'):
        env.unwrapped.turbulence_power = 1.5
    if hasattr(env.unwrapped, 'continuous'):
        env.unwrapped.continuous = False

    # Create the DQN agent
    agent = DQNAgent(env)

    # Train the agent
    num_episodes = 300  # Number of episodes to train the agent
    rewards = agent.train_agent(num_episodes)

    # Display training results
    agent.plot_training_progress(rewards, num_episodes)

    # Close the environment
    env.close()


if __name__ == "__main__":
    main()
