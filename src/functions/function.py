import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def mean_squared_error(y, yHat):
    return np.sum(0.5*(y - yHat)**2)
    
