"""
Functions for visualizing loss during training.  
"""

# pylint: disable=E1101, C0200
from matplotlib import pyplot as plt
import seaborn as sns
import autograd.numpy as np


def print_loss(
    iteration: int,
    loss: float,
    prev_loss: float,
):
    """Prints the iteration number and loss of the neural networks.

    Args:
        t: The input vector
        neural_networks: A list of neural networks
        weights_list: The weights and biases of the neural networks
        iteration: The current iteration
        prev_loss: The loss of of the previous iteration

    Returns:
        loss: The loss of the neural networks
    """
    print("\033[0m", end="")
    print("Iteration: ", iteration)
    if loss < prev_loss:
        print("\033[38;5;121m", loss)
    else:
        print("\033[38;5;210m", loss)

    return loss


def plot_loss(num_iter, loss):
    """Plots the loss of the neural networks.

    Args:
        num_iter: The number of iterations
        loss: np.array of the loss at each iteration

    Returns:
        None
    """
    fig = plt.figure(figsize=(20, 5))
    ax = fig.add_subplot()
    sns.set_theme(style="darkgrid", palette="muted", font="DeJavu Serif")
    ax.plot(np.arange(0, num_iter), loss[:num_iter], lw=1)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    sns.despine()
