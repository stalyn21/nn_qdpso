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
- Opencv-python (CV2)
- Joblib
- Pickle
- Seaborn

To install the requirements execute the [req](https://github.com/stalyn21/nn_qdpso/blob/main/req.txt) file or execute the following command:
```
$ pip install -r req.txt
```
If you have problem with the scikit-learn package, use this command:
```
$ pip install -U scikit-learn
```

## PSO algorithm
The PSO algorithm was proposed in 1995 by Kennedy and Eberhart. This is a metaheuristic algorithm that is based on animal behavior. In this case, The flock of birds inspires PSO. Moreover, this algorithm finds global minimum or maximum. Thus, the PSO algorithm is focus on optimization problems.

There are several variations of this algorithm to optimize a function with one or multiple variables, and these always follow the following steps:

1.- Create an initial swarm of n random particles. Each particle consists of 4 elements: a position that represents a specific combination of variable values, the value of the objective function at the place where the particle is located, a velocity that indicates how and where the particle moves, and a record of the best position in which the particle has been so far.

2.- Evaluate each particle with the objective function.

3.- Update the position and speed of each particle. In this part, the algorithm optimizes the objective function, moving the particle. Hence, the algorithm uses the following equations:
(Flight and Position Function)

4.- If a stop criterion is not met, return to step 2.

## Script usage for image classification
To run the algorithm use the [Multi-class Weather dataset](https://www.kaggle.com/datasets/somesh24/multiclass-images-for-weather-classification), the [src/main.py](https://github.com/stalyn21/nn_qdpso/blob/main/src/main.py) file and execute the following command: 
```
$ python3 main.py
```

In the [main](https://github.com/stalyn21/nn_qdpso/blob/main/src/main.py) file, you can change the images size, alfa or beta value, the particles number, the max iterations and another parameters, see below.
```
# At the beginnig in global variables section
n_particulas = 100
max_iter = 1000
n_training = 10

IMG_HEIGHT = 150
IMG_WIDTH = 150
# uncomment to reduction feature
#dimension = 14

# At the middle in feature reduction section
# Uncomment to reduction feature
#embedding = Isomap(n_components=dimension, n_neighbors=100)
#X_transformed = embedding.fit_transform(X_train_aux)
#print ('Shape of X_train transformed: ', X_transformed.shape)

# At the ending in variables for qdpso section
beta = 1.13
```


