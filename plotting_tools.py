"""
Tools for plotting training results
"""

# pylint: disable=E1101, C0200
import autograd.numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def plot_ode(
    t: np.array,
    res: np.ndarray,
    an_sol: np.ndarray,
    scatter_trial=False,
    scatter_analytical=False,
):
    """Plots the trial solution of the neural network against the analytical solution for ordinary differential equations.

    Args:
        t: input vector
        res: trial solution of the neural network with dim(1, len(t))
        an_sol: analytical solution of the ODE
        scatter_trial: whether to plot the trial solution as a scatter plot
        scatter_analytical: whether to plot the analytical solution as a scatter plot

    Returns:
        None
    """

    sns.set_theme(style="darkgrid", palette="muted", font="DeJavu Serif")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()

    if scatter_trial:
        ax.scatter(t, res[0], lw=1)
    else:
        ax.plot(t, res[0], lw=1)
    if scatter_analytical:
        ax.scatter(t, an_sol, lw=1)
    else:
        ax.plot(t, an_sol, lw=1)

    sns.despine()
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\Psi$")
    plt.legend(["NN", "Analytical"])


def plot_system_ode(
    res: np.ndarray,
    an_sol: np.ndarray,
    scatter_trial=False,
    scatter_analytical=False,
):
    """Plots the trial solution of the neural network against the analytical solution for system of two ordinary differential equations.

    Args:
        res: trial solution of the neural network
        an_sol: analytical solution of the ODE
        scatter_trial: whether to plot the trial solution as a scatter plot
        scatter_analytical: whether to plot the analytical solution as a scatter plot

    Returns:
        None
    """

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
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection="3d")

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
        ax.set_box_aspect(aspect=None, zoom=0.9)

    sns.despine()
    plt.legend(["NN", "Analytical"])


def print_error(res: np.ndarray, an_sol: np.ndarray):
    """Print difference between trial solution and analytical solution.

    Args:
        res: trial solution
        an_sol: analytical solution

    Returns:
        None
    """
    print(res - an_sol)
