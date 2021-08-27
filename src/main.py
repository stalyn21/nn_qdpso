#import module/library
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

from qpso import QDPSO
from sklearn.datasets import load_iris
from functions.function import sigmoid, mean_squared_error

#----------------------------------------
#Generate data
#----------------------------------------

"""data = load_iris()
X = data.data
y = data.target
y = np.array(y).reshape((len(y),1))
"""
X = np.array([[3,5],[5,1],[10,2]])
y = np.array([[.98],[.74],[.57]])

class Neural_Network(object):
    def __init__(self):
        #define network topology
        self.inputLayerSize = 2
        self.hiddenLayerSize_1 = 3
        self.outputLayerSize = 1

        #bias (parameters)
        self.b1 = np.random.randn(self.hiddenLayerSize_1,)
        self.b2 = np.random.randn(self.outputLayerSize,)

        #weight (parameters)
        self.W1 = np.random.randn(self.inputLayerSize, \
                                    self.hiddenLayerSize_1)
        self.W2 = np.random.randn(self.hiddenLayerSize_1, \
                                    self.outputLayerSize)

    def forward(self, X, *p):
        #build topology
        if len(p)==0:
            W1 = self.W1
            b1 = self.b1
            W2 = self.W2
            b2 = self.b1
        else:
            p = np.array(p).reshape((self.NDim,1))
            W1 = p[0:6].reshape((self.inputLayerSize, \
                                    self.hiddenLayerSize_1))
            b1 = p[6:9].reshape((self.hiddenLayerSize_1,))
            W2 = p[9:12].reshape((self.hiddenLayerSize_1, \
                                        self.outputLayerSize))
            b2 = p[12:13].reshape((self.outputLayerSize,))

        #propagate the inputs through network
        self.z2 = np.dot(X, W1) + b1
        self.a2 = sigmoid(self.z2)
        self.z3 = np.dot(self.a2, W2) + b2
        y_hat = sigmoid(self.z3)
        
        return y_hat
    
    def predict(self, pos):        
        y_hat = self.forward(X, pos)
        pred = mean_squared_error(y, y_hat)
        return y_hat, pred

    def objective_function(self, p):
        yHat = self.forward(X, p)
        loss_function = mean_squared_error(y, yHat)
        return loss_function

    def f(self, particle):
        return self.objective_function(particle)

    def log(self, s):
        best_value = [p.best_value for p in s.particles()]
        best_value_avg = np.mean(best_value)
        best_value_std = np.std(best_value)

    def train(self, X, y):
        #initalize parameters for qdpso        
        NParticle = 25
        MaxIters = 100
        self.NDim = 13
        bounds = [(-1, 1) for i in range(0, self.NDim)]
        g = 0.96

        s = QDPSO(self.f, NParticle, self.NDim, bounds, MaxIters, g)
        s.update(callback=self.log, interval=1)
        #print("Found best position: {0}".format(s.gbest))

        y_pred, cost_pred = self.predict(s.gbest)

        return y_pred, cost_pred

nn = Neural_Network()
#loss = nn.predict_forward(X, y)
#print(loss)

y_p1 = nn.forward(X)
cost_p1 = mean_squared_error(y, y_p1)

print("Prediction: ", y_p1)
print("Pediction Cost: ", cost_p1)

y_p, cost_p = nn.train(X, y)

print("Prediction with QDPSO: ", y_p)
print("Pediction Cost: ", cost_p)