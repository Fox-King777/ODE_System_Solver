# ODE System Solver
Implements unsupervised learning neural networks to solve ordinary differential equatios or systems of ordinary differential equations.
Provides tools for plotting results and train loss.

Algorithms can be found in this [paper](https://arxiv.org/pdf/physics/9705023)

## How the algorithm works
The program is only given the differential equation(s) and the initial/boundary conditions. From there the neural network must learn the solution. We want to make sure that the neural newtork is always able to satisfy the initial/boundary conditions and we do this through a *trial solution*. Instead of having the neural newtork directly output predictions of the solution, we incorporate the neural network into another equation to output the predictions. This equation is the *trial solution*, whose sole purpose is to satisfy initial/boundary conditions.

Here is an example

$\frac{dy}{dx} = \gamma y, y_0 = 10$

To satisfy the initial condition, a possible trial solution can be:

$y_t = y_0 + xN(x)$

where N represents the neural newtwork

Because the neural network is not trained labeled data(as there are no labeled data for this problem), the neural network cannot be trained in by convential supervised learning. Instead of having a static target for which the neural network tries to fit, the neural network generates it's own target at each iteration loop. 

Specifically, the neural network computes two values: the derivative of the trail solution and the target derivative


## How it was made
**Libraries used:** Numpy, Matplotlib, Seaborn, Autograd

## Lesson Learned
