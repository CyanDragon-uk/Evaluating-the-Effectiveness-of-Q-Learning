import torch
import torch.nn as nn
import torch.nn.functional as func


class DQN(nn.Module):
    """
    Deep Q-Network (DQN) class using PyTorch.

    This neural network is designed to approximate the Q-function for a
    reinforcement learning problem.
    """

    def __init__(self, input_dim: int, output_dim: int):
        """
        Initialize the DQN model.

        :param input_dim: The dimension of the input state.
        :param output_dim: The dimension of the output action.
        """
        super(DQN, self).__init__()

        # Define the first fully connected layer
        self.fc1 = nn.Linear(input_dim, 128)

        # Define the second fully connected layer
        self.fc2 = nn.Linear(128, 128)

        # Define the third fully connected layer (output layer)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DQN.

        :param x: A batch of input states.
        :return: A batch of Q-values for each action.
        """

        # Apply ReLU activation to the first layer
        x = func.relu(self.fc1(x))

        # Apply ReLU activation to the second layer
        x = func.relu(self.fc2(x))

        # Output layer does not have an activation function
        return self.fc3(x)
