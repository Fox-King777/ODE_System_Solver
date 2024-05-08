"""
Neural Network functions.

"""

# pylint: disable=E1101, C0200
from typing import List, Callable
import autograd.numpy as np


def init_weights(input_size: int, hidden_sizes: np.array, output_size: int):
    """Initializes the weights and biases of the neural network
    Args:
        input_size: Size of the input layer.
        hidden_sizes: List of integers representing the sizes of the hidden layers.
        output_size: Size of the output layer.

    Returns:
        List of weights and biases for each layer of the neural network.
    """
    weights = [None] * (hidden_sizes.shape[0] + 1)  # +1 for the output
    # hidden weights and biases
    weights[0] = np.random.randn(hidden_sizes[0], input_size + 1)  # +1 for the bias
    for i in range(1, hidden_sizes.shape[0]):
        weights[i] = np.random.randn(
            hidden_sizes[i], hidden_sizes[i - 1] + 1
        )  # +1 for the bias

    # output weights and biases
    weights[-1] = np.random.randn(output_size, hidden_sizes[-1] + 1)  # +1 for the bias

    return weights


def forward(
    t: np.array, weights: List[np.array], activation_fns: List[Callable]
) -> np.array:
    """Makes a forward pass through the neural network.

    Args:
        t: The t vector
        weights: The weights and biases of the neural network
        activation_fns: List of activation functions for each layer.

    Returns:
        A NumPy array of the output of the neural network of dim(self.output_size, len(t)).
    """
    num_layers = len(weights)
    # row matrix
    t = t.reshape(-1, t.size)

    z = None
    a = t
    for i in range(num_layers):
        z = np.matmul(weights[i], np.concatenate((np.ones((1, t.size)), a), axis=0))
        a = activation_fns[i](z)
    return z
