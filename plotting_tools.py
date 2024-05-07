"""
Tools for plotting training results
"""

# pylint: disable=E1101, C0200
from typing import List, Callable
import autograd.numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from neural_network import MLPNeuralNetwork


def plot_ode(
    neural_network: MLPNeuralNetwork,
    trial_solution,
    analytical_solution: Callable,
    t: np.array,
):
    """Plots the trial solution of the neural network against the analytical solution for ordinary differential equations.

    Args:
        neural_network: the trained neural network
        analytical_solution: the analytical solution of the DE
        t: input vector

    Returns:
        None
    """
    res = trial_solution(t, neural_network, neural_network.weights)

    sns.set_theme(style="darkgrid", palette="muted", font="DeJavu Serif")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    ax.plot(t, res, lw=1)
    ax.plot(t, analytical_solution(t), lw=1)
    sns.despine()
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\Psi$")
    plt.legend(["nn", "analytical"])


def plot_system_ode(
    neural_networks: List[MLPNeuralNetwork],
    trial_solution,
    analytical_solution: Callable,
    t: np.array,
):
    """Plots the trial solution of the neural network against the analytical solution for system of two ordinary differential equations.

    Args:
        neural_network: the trained neural network
        analytical_solution: the analytical solution of the DE
        t: input vector

    Returns:
        None
    """
    res = [
        trial_solution(t, neural_networks[i], neural_networks[i].weights, i)
        for i in range(len(neural_networks))
    ]

    sns.set_theme(style="darkgrid", palette="muted", font="DeJavu Serif")
    if len(res) == 2:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot()
        ax.plot(res[0], res[1], lw=1)
        ax.plot(analytical_solution(t)[0], analytical_solution(t)[1], lw=1)
        ax.set_xlabel(r"$\Psi_1$")
        ax.set_ylabel(r"$\Psi_2$")
    if len(res) == 3:
        fig = plt.figure(figsize=(10, 10), projection="3d")
        ax = fig.add_subplot()
        ax.plot(res[0], res[1], res[2], lw=1)
        ax.plot(
            analytical_solution(t)[0],
            analytical_solution(t)[1],
            analytical_solution[2],
            lw=1,
        )
        ax.set_xlabel(r"$\Psi_1$")
        ax.set_ylabel(r"$\Psi_2$")
        ax.set_zlabel(r"$\Psi_3$")
    sns.despine()
    plt.legend(["MLP", "Analytical"])
