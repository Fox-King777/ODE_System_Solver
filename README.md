# ODE System Solver
Implements unsupervised learning neural networks to solve ordinary differential equatios or systems of ordinary differential equations.
Provides tools for plotting results and train loss.

Algorithms can be found in this [paper](https://arxiv.org/pdf/physics/9705023).

## How the algorithm works
The program is only given the differential equation(s) and the initial/boundary conditions. From there the neural network must learn the solution. We want to make sure that the neural newtork is always able to satisfy the initial/boundary conditions and we do this through a *trial solution*. Instead of having the neural newtork directly output predictions of the solution, we incorporate the neural network into another equation to output the predictions. This equation is the *trial solution*, whose sole purpose is to satisfy initial/boundary conditions.

Let's look at an exmaple:

$\frac{dy}{dx} = \gamma y, y_0 = 10$

To satisfy the initial condition, a possible trial solution can be:

$y_t = y_0 + xN(x)$,

where N represents the neural newtwork.
At $x=0$, the $xN(x)$$ term evaluates to 0, leaving you with $y_0$. This means at $x=0$, the initial condition of y_0 is met and the trial solution did it's job.

Next is the problem of training. Because the neural network is not trained labeled data(as there are no labeled data for this problem), the neural network cannot be trained in by convential supervised learning. Instead of having a static target for which the neural network tries to fit, the neural network generates it's own target at each iteration loop. 

Specifically, the neural network computes two values: the current derivative and the target derivative. Following the above example:

The current derivative is computed by taking the derivative of the trial solution: cur = $\frac{dy_t}{dx}$
The target derivative is computed by taking the output from the trial solution and inputing it into the right hand side of the differential equation: target = $\gamma y_t$

Simply put, the current derivative is the derivative of the current trial solution and the target derivative is the right hand side of the differential equation. With these values the loss becomes:

$RMSD = \sqrt{\frac{\sum_{i=1}^{N} (target_i - cur_i)^2\}{N}}$

$RMSD$: rooted mean squared error

$i$: iterator over the elements

$N$: number of elements

The neural network is trained with a bounded range of input with this loss function which does not need labeled data. While the given examples worth with just ODEs, this algorithm is generalized for systems of ODEs as well.

## How it was made
**Libraries used:** Numpy, Matplotlib, Seaborn, Autograd

Due to the unorthodox training algorithm, no ML libraries could be utilized. Instead, the core of the algorithm was implemented with Numpy and Autograd libraries. Numpy handled the linear algebra and Autograd handled the math behind automatic differentiation. The core of the algorithm was split into activation functions, loss function, neural network, and optimizers. The activation functions file includes the sigmoid, tanh, arctan, and ELU activation functions. The loss function file includes the RMSE loss as well as the Autograd automatic differentian function needed for the loss calculation. The optimizers file includes implementations of gradient descent as well as Adam, which are used for training the models. In both optimizers, Autograd automatic differentiation is used for backward propagation.

Due to the limitations of the Autograd library, it is only able to differentiate functions with respect to one of the function parameters. This means classes could not be ultilized. As such, the neural network file only includes weight initilization and forward propogation instead of a fully wrapped neural network class. Futhermore, activation functions, weights, and trial solutions are passed into forward propogation, optimizers, and more separately.

Additionally, my project includes tools for plotting results with Matplotlib and Seaborn, and tools for analyzing training including the colorization of loss based on increases or decrease and plotting losss.

The initialization and training of the neural networks are done in jupyter notebooks, with each notebook solving a different ODE or system of ODEs.

## Optimization
Initially, the only training method I implemented was gradient descent. However, with gradient descent I started to notice that the loss during training was very volatile. Training often resulted in NaN values, or Not a Number values, meaning that the loss grew way too large. I had to set the learning rate extremely low to battle this. However, with such a small learning rate, I was unable to see the results of my models because it was way too slow. After implementing the Adam optimizer, which has momentum and bias correction, the training ran much more stable and fast. For most of my experiments, the loss started to converge and the resulting graph was very close to the expected outcome. The Lorenz system is an outlier to my experiments with the neural network not even able to get close to the right results. Because Lorenz system is a choatic system, another optimizer is needed, on that is able to converge the loss to an extreme precision.

## Lesson Learned
This is one of the most challenging projects I have worked on but also one I learned the most from. Having no other choice than to implement a model from scratch taught me a lot about the math and specific details of neural networks. The unique loss function also serves as a reminder that neural networks much more flexible in the way they can be trained. This project also serves as a reminder that neural networks are not good at every problem, and may fall short when it comes to chaotic/random problems.