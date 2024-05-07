"""
Neural Network designed to solve DEs and syste of DEs.

"""

# pylint: disable=E1101, C0200
from typing import List, Callable
import autograd.numpy as np


class MLPNeuralNetwork:
    """A general MLP neural network class.
    Attributes:
        input_size (int): Size of the input layer.
        hidden_sizes (np.array): Array of integers representing the sizes of the hidden layers.
        output_size (int): Size of the output layer.
        activation_fns (List[Callable]): List of activation functions for each layer.
        weights (List[np.array]): List of weights and biases for each layer.

    Methods:
        init_weights(): Initializes the weights and biases of the neural network.
        forward(t: np.array, weights: List[np.array]) -> np.array: Makes a forward pass through the neural network.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: np.array,
        output_size: int,
        activation_fns: List[Callable],
    ):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation_fns = activation_fns
        self.weights = [None] * (hidden_sizes.shape[0] + 1)  # +1 for the output
        self.init_weights()

    def init_weights(self):
        """Initializes the weights and biases of the neural network
        Args:
            None

        Returns:
            None
        """
        # hidden weights and biases
        self.weights[0] = np.random.randn(
            self.hidden_sizes[0], self.input_size + 1
        )  # +1 for the bias
        for i in range(1, self.hidden_sizes.shape[0]):
            self.weights[i] = np.random.randn(
                self.hidden_sizes[i], self.hidden_sizes[i - 1] + 1
            )  # +1 for the bias

        # output weights and biases
        self.weights[-1] = np.random.randn(
            self.output_size, self.hidden_sizes[-1] + 1
        )  # +1 for the bias

    def forward(self, t: np.array, weights: List[np.array]) -> np.array:
        """Makes a forward pass through the neural network.

        Args:
            t: The t vector
            weights: The weights and biases of the neural network

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
            a = self.activation_fns[i](z)
        return z
