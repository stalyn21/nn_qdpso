#--------------
#import modules
#--------------

import numpy as np
from timeit import default_timer

from sklearn.metrics import mean_squared_error
from qpso.qpso import QDPSO

#-------------------------
#Activation function class
#-------------------------

class activationFunction(object):
    def __init__(self, X_sample):
        self.n_sample_af = X_sample

    def softmax(self, z):
        #z -= np.max(z)
        exps = np.exp(z)
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    def sigmoid(self, z):
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
        corect_logprobs = -np.log(y_hat[range(self.n_sample_lf), y])
        return np.sum(corect_logprobs) / self.n_sample_lf
    
    def crossEntropy(self, y, y_hat):
        ind_loss = np.max(-1 * y * np.log(y_hat + 1e-12), axis=1)
        Ls_ce = np.sum(ind_loss) / self.n_sample_lf
        L2 = self.beta/self.n_sample_lf
        return Ls_ce + L2
    """

    def crossEntropy(self, y, y_hat, weights):
        ind_loss = np.max(-1 * y * np.log(y_hat + 1e-12), axis=1)
        Ls_ce = np.sum(ind_loss) / self.n_sample_lf
        # L2 regularization term
        L2 = (self.beta / (2 * self.n_sample_lf)) * np.sum([np.sum(np.square(w)) for w in weights])
        return Ls_ce + L2
    """

    def mse(self, y, y_hat):
        return np.sum(0.5*(y - y_hat)**2)

#-----------------
# Perceptron Class
#-----------------

class perceptron(object):
    def __init__(self, X_sample, X_input, X_class):
        # initialize parameters
        #np.random.seed(444)
        self.n_inputs = X_input
        self.n_hidden = X_input * 3
        self.n_outputs = X_class
        self.n_sample = X_sample

        self.timings = []    

        self.af = activationFunction(X_sample)
        self.lf = lossFunction(X_sample)       

        self.d = (self.n_inputs * self.n_hidden) + (self.n_hidden * self.n_outputs) + self.n_hidden + self.n_outputs

        # Weight and Bias parameters
        self.W1 = np.random.randn(self.n_inputs, self.n_hidden)
        self.W2 = np.random.randn(self.n_hidden, self.n_outputs)
        self.b1 = np.random.randn(self.n_hidden, )
        self.b2 = np.random.randn(self.n_outputs, )

        # parameters random initialization
        self.param = np.random.randn(self.d, )
    
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
        return loss

    def f(self, x):
        j = self.backware(x)
        return j
    
    def log(self, s):
        best_value = [p.best_value for p in s.particles()]
        best_value_avg = np.mean(best_value)
        best_value_std = np.std(best_value)
        y_hat_pred_hidden = np.argmax(self.forward(self.X_test, s.gbest), axis=1)
        cost_pred_hidden = mean_squared_error(self.y_test, y_hat_pred_hidden)        
        # Update the best parameters of the hidden layer
        if (cost_pred_hidden <= self.error_hidden):
            self.error_hidden = cost_pred_hidden
            self.param = s.gbest
        self.g_best_value_iter.append(s.gbest_value)
        self.avg_best_value.append(best_value_avg)
        self.std_best_value.append(best_value_std)  
        print("{0: >5}  {1: >9.3E}  {2: >9.3E}  {3: >9.3E}".format(s.iters, s.gbest_value, best_value_avg, best_value_std))
        print("Current hidden error: ", cost_pred_hidden)
        print("Best hidden error: ", self.error_hidden)        

    def one_hot_encode(self, Y):
        num_unique = len(np.unique(np.array(Y)))
        zeros = np.zeros((len(Y), num_unique))
        zeros[range(len(Y)), Y] = 1
        return zeros

    def train(self, X, y, X_test, y_test, beta, n_particulas, max_iter, i):
        self.g_best_value = []
        self.g_best_value_iter = []
        self.avg_best_value = []
        self.std_best_value = []

        print(">> Training nn with QDPSO")
        self.X = X
        self.y = self.one_hot_encode(y)        
        self.error_hidden = 100
        self.error_output = 100
        self.X_test = X_test
        self.y_test = y_test

        #initalize parameters for qdpso 
        NParticle = n_particulas
        MaxIters = max_iter
        bounds = [(-1, 1) for _ in range(0, self.d)]
        g = beta #1.7 #.96

        start_time_training = default_timer()
        print("> Training layer")
        s = QDPSO(self.f, NParticle, self.d, bounds, MaxIters, g)
        s.update(callback=self.log, interval=1)        
        end_time_training = default_timer()
        self.timings.append((f"total_training_{i}", start_time_training, end_time_training))

        self.g_best_value = np.array(s.gbest_value)
        self.g_best_value_iter = np.array(self.g_best_value_iter)
        self.avg_best_value = np.array(self.avg_best_value)
        self.std_best_value = np.array(self.std_best_value)
        return self.param