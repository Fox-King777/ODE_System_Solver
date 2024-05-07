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
    scatter_trial=False,
    scatter_analytical=False,
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
    an_sol = analytical_solution(t)

    sns.set_theme(style="darkgrid", palette="muted", font="DeJavu Serif")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()

    if scatter_trial:
        ax.scatter(t, res, lw=1)
    else:
        ax.plot(t, res, lw=1)
    if scatter_analytical:
        ax.scatter(t, an_sol, lw=1)
    else:
        ax.plot(t, an_sol, lw=1)

    sns.despine()
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\Psi$")
    plt.legend(["nn", "analytical"])


def plot_system_ode(
    neural_networks: List[MLPNeuralNetwork],
    trial_solution,
    analytical_solution: Callable,
    t: np.array,
    scatter_trial=False,
    scatter_analytical=False,
):
    """Plots the trial solution of the neural network against the analytical solution for system of two ordinary differential equations.

    Args:
        neural_network: the trained neural network
        analytical_solution: the analytical solution of the DE
        t: input vector

    Returns:
        None
    """
    res = np.array(
        [
            trial_solution(t, neural_networks[i], neural_networks[i].weights, i)
            for i in range(len(neural_networks))
        ]
    )
    an_sol = analytical_solution(t)

    sns.set_theme(style="darkgrid", palette="muted", font="DeJavu Serif")

    if len(res) == 2:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot()

        if scatter_trial:
            ax.scatter(res[0], res[1], lw=1)
        else:
            ax.plot(res[0], res[1], lw=1)
        if scatter_analytical:
            ax.scatter(an_sol[0], an_sol[1], lw=1)
        else:
            ax.plot(an_sol[0], an_sol[1], lw=1)

        ax.set_xlabel(r"$\Psi_1$")
        ax.set_ylabel(r"$\Psi_2$")

    if len(res) == 3:
        fig = plt.figure(figsize=(10, 10), projection="3d")
        ax = fig.add_subplot()

        if scatter_trial:
            ax.scatter(res[0], res[1], res[2], lw=1)
        else:
            ax.plot(res[0], res[1], res[2], lw=1)
        if scatter_analytical:
            ax.scatter(an_sol[0], an_sol[1], an_sol[2], lw=1)
        else:
            ax.plot(an_sol[0], an_sol[1], an_sol[2], lw=1)

        ax.set_xlabel(r"$\Psi_1$")
        ax.set_ylabel(r"$\Psi_2$")
        ax.set_zlabel(r"$\Psi_3$")

    sns.despine()
    plt.legend(["MLP", "Analytical"])


def print_error(res: np.ndarray, an_sol: np.ndarray):
    """Print difference between trial solution and analytical solution.

    Args:
        res: trial solution
        an_sol: analytical solution

    Returns:
        None
    """
    print(res - an_sol)
