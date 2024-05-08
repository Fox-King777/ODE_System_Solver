"""
Optimizers for neural network training
"""

# pylint: disable=E1101, C0200
from typing import List, Callable
import autograd.numpy as np
from autograd import grad
from loss_function import mse_loss_function
from training_utility import print_loss, plot_loss


def gradient_descent(
    t: np.array,
    weights_list: List[List[np.array]],
    trial_solution: Callable,
    activation_fns: List[Callable],
    derivative: Callable,
    threshold=0.01,
    num_iters=10000,
    learn_rate=0.0001,
    log=True,
    plot=True,
):
    """Runs gradient descent for a given number of iterations.

    Args:
        t: The input vector
        weights_list: A list of weights for each neural network
        activation_fns: List of activation functions for each layer
        trial_solution: The trial solution function
        derivative: The true derivative of the differential equation
        threshold: The threshold for stropping gradient descent
        num_iters: The max number of iterations
        learn_rate: The learning rate
        log: Whether to print the loss
        plot: Whether to plot the loss

    Returns:
        weights_list: A list of weights for each neural network after running gradient descent
    """
    loss_grad_function = grad(mse_loss_function, 1)
    loss = [0] * num_iters
    for i in range(num_iters):
        loss[i] = mse_loss_function(
            t, weights_list, activation_fns, trial_solution, derivative
        )

        if log is True:
            print_loss(i, loss[i], loss[i - 1])
        if loss[i] < threshold:
            num_iters = i + 1
            break

        loss_grad = loss_grad_function(
            t, weights_list, activation_fns, trial_solution, derivative
        )
        for j in range(len(weights_list)):
            for k in range(len(weights_list[j])):
                weights_list[j][k] = weights_list[j][k] - learn_rate * loss_grad[j][k]

    if plot is True:
        plot_loss(num_iters, loss)

    return weights_list


def adam(
    t: np.array,
    weights_list: List[List[np.array]],
    activation_fns: List[Callable],
    trial_solution: Callable,
    derivative: Callable,
    num_iters=10000,
    step_size=0.001,
    threshold=0.01,
    b1=0.9,
    b2=0.999,
    eps=10**-8,
    log=True,
    plot=True,
):
    """Runs Adam for a given number of iterations.

    Args:
        t: The input vector
        weights_list: A list of weights for each neural network
        activation_fns: List of activation functions for each layer
        trial_solution: The trial solution function
        derivative: The true derivative of the differential equation
        num_iters: The number of iterations
        step_size: The step size per iteration
        b1: The first moment estimate coefficient
        b2: The second moment estimate coefficient
        eps: The epsilon for numerical stability
        log: Whether or not to print loss
        plot: Whether or not to plot loss

    Returns:
        weights_list: A list of weights for each neural network after running Adam
    """
    loss_grad_function = grad(mse_loss_function, 1)
    m = [
        [np.zeros_like(weights_list[i][j]) for j in range(len(weights_list[i]))]
        for i in range(len(weights_list))
    ]
    v = [
        [np.zeros_like(weights_list[i][j]) for j in range(len(weights_list[i]))]
        for i in range(len(weights_list))
    ]

    mhat = None
    vhat = None

    loss = [0] * num_iters
    for i in range(num_iters):
        loss[i] = mse_loss_function(
            t, weights_list, activation_fns, trial_solution, derivative
        )
        if log is True:
            print_loss(i, loss[i], loss[i - 1])
        if loss[i] < threshold:
            num_iters = i + 1
            break

        g = loss_grad_function(
            t, weights_list, activation_fns, trial_solution, derivative
        )
        for j in range(len(weights_list)):
            for k in range(len(weights_list[j])):
                # First  moment estimate.
                m[j][k] = (1 - b1) * g[j][k] + b1 * m[j][k]
                # Second moment estimate.
                v[j][k] = (1 - b2) * (g[j][k] ** 2) + b2 * v[j][k]

                # Bias correction.
                mhat = m[j][k] / (1 - b1 ** (i + 1))
                vhat = v[j][k] / (1 - b2 ** (i + 1))
                weights_list[j][k] = weights_list[j][k] - step_size * mhat / (
                    np.sqrt(vhat) + eps
                )

    if plot is True:
        plot_loss(num_iters, loss)

    return weights_list
