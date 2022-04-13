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

## Script usage for image classification
To run the algorithm use the [Intel dataset](https://drive.google.com/drive/folders/1GQShwbuKDCOCREev573iUhqk9pcFWGs8?usp=sharing), the [nn_qdpso/src/qpso_nn/imageClassification-qpso/main.py](https://github.com/stalyn21/nn_qdpso/blob/main/src/qpso_nn/imageClassification-qpso/main.py) file and execute the following command: 
```
$ python3 main.py
```

In the [main](https://github.com/stalyn21/nn_qdpso/blob/main/src/qpso_nn/imageClassification-qpso/main.py) file, you can change the images size, alfa or beta value, the particles number, the max iterations and the gbest/bestValue name, see below.
```
# At the beginnig
IMG_HEIGHT = 30
IMG_WIDTH = 30 

# At the end
beta = 1.13 #1.7
n_particulas = 25
max_iter = 1000
gBest = nn.train(X_train, y_train, beta, n_particulas, max_iter)
print('best value', gBest)

# The file name is 'gBest/bValue_beta_nParticles_maxIter_imgHEIGHTximgWIDTH_rgbChannel.npy'
# Save global best value
np.save("gBest_113_25_1000_30x30_1.npy", gBest)
# Save best value during the trainning 
np.save("bValue_113_25_1000_30x30_1.npy", nn.metric_loss)
```


