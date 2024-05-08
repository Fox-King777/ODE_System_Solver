"""
Loss function.
"""

# pylint: disable=E1101, C0200
from typing import List, Callable
import autograd.numpy as np
from autograd import elementwise_grad


def mse_loss_function(
    t: np.array,
    weights_list: List[List[np.array]],
    activation_fns: List[Callable],
    trial_solution: Callable,
    derivative: Callable,
) -> float:
    """Calculates the mean squared error a list of neural network.

    Args:
        t: The input vector
        weights_list: A list of weights and biases for each neural networks
        activation_fns: List of activation functions for each layer
        trial_solution: The trial solution function
        derivative: The true derivative of the differential equation

    Returns:
        Mean squared error value
    """
    trial_sol = np.array(
        trial_solution(t, weights_list, activation_fns)
    )  # dim(len(weight_list), len(t))

    grad_star = derivative(t, trial_sol)
    nn_grad = trial_grad(t, weights_list, activation_fns, trial_solution)

    error = grad_star - nn_grad
    loss = error**2 / (error.size * len(weights_list))
    loss = np.sum(loss)
    loss = np.sqrt(loss)

    return loss


def trial_grad(
    t: np.array,
    weights_list: List[List[np.array]],
    activation_fns: List[Callable],
    trial_solution: Callable,
) -> np.ndarray:
    """Calculates the gradient of the trial solution of the Lorentz System.

    Args:
        t: input vector
        weights_list: list of weights and biases for each neural networks
        activation_fns: List of activation functions for each layer
        trial_solution: trial solution function

    Returns:
        A NumPy array of the gradient of the trial solution
        dimension (len(weights_list), len(t))
    """
    return np.array(
        [
            elementwise_grad(elementwise_trial_solution, 0)(
                t, weights_list, activation_fns, trial_solution, i
            )
            for i in range(len(weights_list))
        ]
    )


def elementwise_trial_solution(
    t: np.array,
    weights_list: List[List[np.array]],
    activation_fns: List[Callable],
    trial_solution: Callable,
    idx: int,
) -> np.array:
    """Calculates one of the outputs of the trial solution given an index.

    Args:
        t: input vector
        weights_list: list of weights and biases for each neural networks
        activation_fns: List of activation functions for each layer
        trial_solution: trial solution function
        idx: index

    Returns:
        Trial solution at the given index
    """
    return trial_solution(t, weights_list, activation_fns)[idx]
