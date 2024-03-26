# Senior-Project
Everything around us, from biology, stocks, physics, or even common life scenarios can be mathematically modeled by differential equations. From exponential growth/decay and simple harmonic motion equation to the Laplace equation, used to describe steady state heat conduction, and Navier-Stokes equation, used to describe fluid turbulent flow, differential equations can range from easy to solve to incredibly difficult. This project with be focused on solving systems of ordinary differential equations(ODEs). Traditional methods used to solve these systems of ODEs such as the Euler method have truncation and round-off errors as well as instability with stiff functions, functions with flat areas, that make it less accurate for higher order equations while methods such as Runge-Kutta, RDK4, is accurate to the fourth power but is 4 times as costly as the Euler method. This is where neural networks come in. Neural networks have an advantage over traditional methods because the non-linearity of these systems do not cause problems as the would for traditional methods, such as the stiff functions discussed above. Additionally, neural networks can handle high dimensionality with ease, which makes it perfect for large amounts of data with multiple interacting variables. Because neural networks are not slowed down by non-linearity and high dimensionality, they can predict systems of ODEs more accurately and faster than traditional methods. Through research and experimentation, I will create a neural network model using Python libraries to predict the Lorentz System, a chaotic system in which switching the initial conditions can drastically change the function.