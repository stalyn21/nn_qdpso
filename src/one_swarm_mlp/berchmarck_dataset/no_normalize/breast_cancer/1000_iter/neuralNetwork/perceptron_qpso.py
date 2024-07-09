#--------------
#import modules
#--------------

import numpy as np

from numpy import linalg as LA
from qpso.qpso import QDPSO

#-------------------------
#Activation function class
#-------------------------

class activationFunction(object):
    def __init__(self, X_sample):
        self.n_sample_af = X_sample

    def softmax(self, z):
        """
        Apply softmax function on logits and return probabilities.
        :param logits: double(N, C)
        Logits of each instance for each class.
        :return: double(N, C)
        probability for each class of each instance.
        """
        exps = np.exp(z)
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    def sigmoid(self, z):
        #sig = np.where(z<0, np.exp(z)/(1 + np.exp(z)), 1/(1 + np.exp(-z)))
        #return sig
        """#Numerically stable sigmoid function.
        if x >= 0:
            z = exp(-x)
            return 1 / (1 + z)
        else:
            # if x is less than zero then z will be small, denom can't be
            # zero because it's 1+z.
            z = exp(x)
            return z / (1 + z)"""
        return np.exp(-np.logaddexp(0, -z))
    
    def relu(self, z):
        return np.maximum(z, 0)

#--------------------
#Loss functions class
#--------------------

class lossFunction(object):
    def __init__(self, X_sample):
        self.n_sample_lf = X_sample
        self.beta = 5e-4

    def NLL(self, y, y_hat):
        #correct_logprobs = - np.log(z)
        #return np.sum(correct_logprobs) / self.n_sample_lf
        corect_logprobs = -np.log(y_hat[range(self.n_sample_lf), y])
        return np.sum(corect_logprobs) / self.n_sample_lf
    
    def crossEntropy(self, y, y_hat):
        """
        Calculates Categorical Cross Entropy loss.
        :param probs: double(N, C)
            Probability of each instance for each class.
        :param Y: int(N, C)
            One-hot encoded representation of classes.
        :return: double
            Returns value of loss calculated.
        """
        #num_samples = len(y_hat)
        ind_loss = np.max(-1 * y * np.log(y_hat + 1e-12), axis=1)
        Ls_ce = np.sum(ind_loss) / self.n_sample_lf
        # L2 regularization (penalty)
        # 2 inside of the LA.norm() is equal to 2-norm (largest sing. value)
        """
        try:
            L2 = self.beta * LA.norm(np.subtract(y, y_hat), 2)
        except:
            L2 = self.beta
        """
        L2 = self.beta/self.n_sample_lf
        return Ls_ce + L2

    def mse(self, y, y_hat):
        return np.sum(0.5*(y - y_hat)**2)

#-----------------
# Perceptron Class
#-----------------

class perceptron(object):
    def __init__(self, X_sample, X_input, X_class):
        # initialize parameters
        self.n_inputs = X_input
        self.n_hidden = X_input * 3
        self.n_outputs = X_class
        self.n_sample = X_sample
        
        self.af = activationFunction(X_sample)
        self.lf = lossFunction(X_sample)
        self.g_best_value_iter = []
        self.avg_best_value = []
        self.std_best_value = []
        self.d = (self.n_inputs * self.n_hidden) + (self.n_hidden * self.n_outputs) + self.n_hidden + self.n_outputs

        # Weight and Bias parameters
        self.W1 = np.random.randn(self.n_inputs, self.n_hidden)
        self.W2 = np.random.randn(self.n_hidden, self.n_outputs)
        self.b1 = np.random.randn(self.n_hidden, )
        self.b2 = np.random.randn(self.n_outputs, )
    
    def forward(self, X, *p):
        if (len(p)==0):
            W1 = self.W1
            b1 = self.b1
            W2 = self.W2
            b2 = self.b2
            
        else:
            p = np.array(p).reshape((self.d, 1))
            W1_lim = self.n_inputs * self.n_hidden
            W1 = p[0:W1_lim].reshape((self.n_inputs, self.n_hidden))
            b1_lim = W1_lim + self.n_hidden
            b1 = p[W1_lim:b1_lim].reshape((self.n_hidden, ))
            W2_lim = b1_lim + (self.n_hidden * self.n_outputs)
            W2 = p[b1_lim:W2_lim].reshape((self.n_hidden, self.n_outputs))
            b2_lim = W2_lim + self.n_outputs
            b2 = p[W2_lim:b2_lim].reshape((self.n_outputs, ))

        z1 = np.dot(X, W1) + b1
        a1 = np.tanh(z1)
        z2 = np.dot(a1, W2) + b2
        y_hat = self.af.softmax(z2)
        return y_hat
    
    def backware(self, p):
        y_hat = self.forward(self.X, p)
        loss = self.lf.crossEntropy(self.y, y_hat)
        #self.metric_loss.append(loss)
        return loss

    def f(self, x):
        j = self.backware(x)
        return j
    
    def log(self, s):
        best_value = [p.best_value for p in s.particles()]
        best_value_avg = np.mean(best_value)
        best_value_std = np.std(best_value)
        self.g_best_value_iter.append(s.gbest_value)
        self.avg_best_value.append(best_value_avg)
        self.std_best_value.append(best_value_std)
        print("{0: >5}  {1: >9.3E}  {2: >9.3E}  {3: >9.3E}".format(s.iters, s.gbest_value, best_value_avg, best_value_std))

    def one_hot_encode(self, Y):
        """
        create one-hot encoded vectors from target labels(Y).
        :param Y: int(N, )
        :return: int(N, C)
            Returns an array of shape(N, C) where C is number of classes.
        """
        num_unique = len(np.unique(np.array(Y)))
        zeros = np.zeros((len(Y), num_unique))
        zeros[range(len(Y)), Y] = 1
        return zeros

    def train(self, X, y, beta, n_particulas, max_iter):
        print("training nn with QDPSO")
        self.X = X
        self.y = self.one_hot_encode(y)
        
        #initalize parameters for qdpso 
        NParticle = n_particulas
        MaxIters = max_iter
        bounds = [(-1, 1) for i in range(0, self.d)]
        g = beta #1.7 #.96

        s = QDPSO(self.f, NParticle, self.d, bounds, MaxIters, g)
        s.update(callback=self.log, interval=1)

        self.g_best_value = s.gbest_value
        self.g_best_value_iter = np.array(self.g_best_value_iter)
        self.avg_best_value = np.array(self.avg_best_value)
        self.std_best_value = np.array(self.std_best_value)
        
        return s.gbest
