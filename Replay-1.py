from collections import deque
import numpy as np
import random
import torch
from typing import Tuple


class ReplayBuffer:
    """
    Replay Buffer for storing and sampling experience tuples.
    """

    def __init__(self, capacity: int):
        """
        Initialize the replay buffer with a specified capacity.

        :param capacity: The maximum number of experiences the buffer can hold.
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """
        Add an experience tuple to the buffer.

        :param state: The state observed.
        :param action: The action taken.
        :param reward: The reward received.
        :param next_state: The next state observed.
        :param done: A boolean indicating if the episode ended.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a batch of experiences from the buffer.

        :param batch_size: The number of experiences to sample.
        :return: A tuple of batched experiences (states, actions, rewards, next_states, dones).
        """
        # Randomly sample a batch of experiences from the buffer
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))

        # Convert the sampled experiences to PyTorch tensors
        return (
            torch.tensor(np.array(state), dtype=torch.float32),
            torch.tensor(action, dtype=torch.long),
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(np.array(next_state), dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32),
        )

    def __len__(self) -> int:
        """
        Return the current size of the buffer.

        :return: The number of experiences currently stored in the buffer.
        """
        return len(self.buffer)
