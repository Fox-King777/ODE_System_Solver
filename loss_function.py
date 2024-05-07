"""
Loss functions
"""

# pylint: disable=E1101, C0200
from typing import List
import autograd.numpy as np
from neural_network import NeuralNetwork


def mse_loss_function(
    t: np.array,
    neural_networks: List[NeuralNetwork],
    weights_list: List[List[np.array]],
) -> float:
    """Calculates the mean squared error a list of neural network.

    Args:
        t: The input vector
        neural_networks: A list of neural networks

    Returns:
        Mean squared error value
    """
    loss = 0
    trial_sol = np.array(
        [
            neural_networks[i].trial_solution(t, weights_list[i])
            for i in range(len(neural_networks))
        ]
    )

    for i in range(len(neural_networks)):
        grad_star = neural_networks[i].derivative(t, trial_sol)
        nn_grad = neural_networks[i].trial_grad(t, weights_list[i])
        error = grad_star - nn_grad
        loss += error**2
    loss /= loss.size
    loss = np.sum(loss)
    loss = np.sqrt(loss)

    return loss
