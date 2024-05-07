"""
Loss function.
"""

# pylint: disable=E1101, C0200
from typing import List, Callable
import autograd.numpy as np
from autograd import elementwise_grad
from neural_network import MLPNeuralNetwork


def mse_loss_function(
    t: np.array,
    neural_networks: List[MLPNeuralNetwork],
    weights_list: List[List[np.array]],
    trial_solution: Callable,
    derivative: Callable,
) -> float:
    """Calculates the mean squared error a list of neural network.

    Args:
        t: The input vector
        neural_networks: A list of neural networks
        weights_list: A list of weights and biases for each neural networks

    Returns:
        Mean squared error value
    """
    loss = 0
    trial_sol = np.array(
        [
            trial_solution(t, neural_networks[i], weights_list[i], i)
            for i in range(len(neural_networks))
        ]
    )

    for i in range(len(neural_networks)):
        grad_star = derivative(t, trial_sol)
        nn_grad = trial_grad(t, neural_networks[i], weights_list[i], trial_solution)
        error = grad_star - nn_grad
        loss += error**2
    loss /= loss.size * len(neural_networks)
    loss = np.sum(loss)
    loss = np.sqrt(loss)

    return loss


def trial_grad(
    t: np.array, nn: MLPNeuralNetwork, weights: List[np.array], trial_solution: Callable
) -> np.array:
    """Calculates the gradient of the trial solution of the Lorentz System.

    Args:
        t: The input vector
        weights: The weights and biases of the neural network

    Returns:
        A NumPy array of the gradient of the trial solution
        dimension (len(t),)
    """
    return elementwise_grad(trial_solution, 0)(t, nn, weights)
