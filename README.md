# Trainning a neural network with QPSO
The project idea is to train a neural network with a quantum-inspired algorithm. In this case, we will use the QPSO (Quantum-behaved Particle Swarm Optimization), which is an improved version of the PSO (Particle swarm optimization). Besides putting it to the test with an application in the field of Artificial Intelligence.

## Requirements
- GNU/Linux (Kali Linux in this case)
- Python 3.9.2
- Pyswarms
- QDPSO
- Scikit-learn 
- Numpy
- Matplotlip

To install the requirements execute the [req](https://github.com/stalyn21/nn_qdpso/blob/main/req.txt) file or execute the following command:
```
$ pip install -r req.txt
```
If you have problem with the scikit-learn package, use this command:
```
$ pip install -U scikit-learn
```

## PSO algorithm
(Make an algorithm description)

There are several variations of this algorithm to optimize a function with one or multiple variables, and these always follow the following steps:

1.- Create an initial swarm of n random particles. Each particle consists of 4 elements: a position that represents a specific combination of variable values, the value of the objective function at the place where the particle is located, a velocity that indicates how and where the particle moves, and a record of the best position in which the particle has been so far.

2.- Evaluate each particle with the objective function.

3.- Update the position and speed of each particle. In this part, the algorithm optimizes the objective function, moving the particle. Hence, the algorithm uses the following equations:
(Flight and Position Function)

4.- If a stop criterion is not met, return to step 2.
