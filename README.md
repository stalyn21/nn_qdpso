# Training an Artificial Neural Network with QPSO
The project idea is to train a neural network with a quantum-inspired algorithm. In this case, we will use the QPSO ([Quantum-behaved Particle Swarm Optimization](https://ieeexplore.ieee.org/document/1330875)), which is an improved version of the PSO (Particle swarm optimization). Besides putting it to the test with an application in the field of Artificial Intelligence.

## Repository Structure
This repository has:
- [Training ANN using one swarm.](./src/one_swarm_mlp/one_swarm_mlp.md)
- Training ANN using multi-swarm.

## PSO algorithm
The PSO algorithm was proposed in 1995 by Kennedy and Eberhart. This is a metaheuristic algorithm that is based on animal behavior. In this case, The flock of birds inspires PSO. Moreover, this algorithm finds global minimum or maximum. Thus, the PSO algorithm is focus on optimization problems.

There are several variations of this algorithm to optimize a function with one or multiple variables, and these always follow the following steps:

1.- Create an initial swarm of n random particles. Each particle consists of 4 elements: a position that represents a specific combination of variable values, the value of the objective function at the place where the particle is located, a velocity that indicates how and where the particle moves, and a record of the best position in which the particle has been so far.

2.- Evaluate each particle with the objective function.

3.- Update the position and speed of each particle. In this part, the algorithm optimizes the objective function, moving the particle. Hence, the algorithm uses the following equations:
(Flight and Position Function)

4.- If a stop criterion is not met, return to step 2.
