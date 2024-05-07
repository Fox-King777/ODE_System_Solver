"""
Provides activation functions for neural networks.    
"""

# pylint: disable=E1101
import autograd.numpy as np


def sigmoid(z: float) -> float:
    """The sigmoid function.

    Args:
        z: input

    Returns:
        sigmoid of z
    """
    return 1 / (1 + np.exp(-z))


def tanh(z: float) -> float:
    """The tanh function.

    Args:
        z: input

    Returns:
        tanh of z
    """
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))


def arctan(z: float) -> float:
    """The arctan function.

    Args:
        z: input

    Returns:
        arctan of z
    """
    return np.arctan(z)


def elu(z: float, alpha=1.0) -> float:
    """The Exponential Linear Unit function.

    Args:
        z: input
        alpha: hyperparameter

    Returns:
        ELU of z
    """
    return np.where(z < 0, alpha * (np.exp(z) - 1), z)
