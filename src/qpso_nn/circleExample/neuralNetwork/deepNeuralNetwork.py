import numpy as np
from qpso import QDPSO

class activationFunction(object):
    def __init__(self):
        self.n_sample_af = 400

    def softmax(self, z):
        # compute softmax values for each sets of score in z.
        #e_z = np.exp(z - np.max(z))
        #return e_z / e_z.sum(axis=0)
        exp_scores = np.exp(z)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    def sigmoid(self, z):
        sig = np.where(z<0, np.exp(z)/(1 + np.exp(z)), 1/(1 + np.exp(-z)))
        return sig 

class lossFunction(object):
    def __init__(self):
        self.n_sample_lf = 400

    def NLL(self, y, y_hat):
        #correct_logprobs = - np.log(z)
        #return np.sum(correct_logprobs) / self.n_sample_lf
        corect_logprobs = -np.log(y_hat[range(self.n_sample_lf), y])
        return np.sum(corect_logprobs) / self.n_sample_lf
    
    def crossEntropy(self, y, y_hat):
        cost = -(1/self.n_sample_lf) * np.sum(y * np.log(y_hat) + (1-y)*np.log(1-y_hat))
        return cost

    def mse(self, y, y_hat):
        return np.sum(0.5*(y - y_hat)**2)

class neuralNetwork(object):
    def __init__(self):
        # initialize parameters
        self.n_inputs = 2
        self.n_hidden = 10
        self.n_outputs = 2
        self.n_sample = 400
        self.af = activationFunction()
        self.lf = lossFunction()
        self.metric_loss = []

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
            W1 = p[0:20].reshape((self.n_inputs, self.n_hidden))
            b1 = p[20:30].reshape((self.n_hidden, ))
            W2 = p[30:50].reshape((self.n_hidden, self.n_outputs))
            b2 = p[50:52].reshape((self.n_outputs, ))

        z1 = np.dot(X, W1) + b1
        a1 = np.tanh(z1)
        z2 = np.dot(a1, W2) + b2
        y_hat = self.af.softmax(z2)
        return y_hat
    
    def backware(self, p):
        y_hat = self.forward(self.X, p)
        loss = self.lf.NLL(self.y, y_hat)
        #self.metric_loss.append(loss)
        return loss

    def f(self, x):
        j = self.backware(x)
        return j

    def predict(self, X, p_best):
        W1 = p_best[0:20].reshape((self.n_inputs, self.n_hidden))
        b1 = p_best[20:30].reshape((self.n_hidden, ))
        W2 = p_best[30:50].reshape((self.n_hidden, self.n_outputs))
        b2 = p_best[50:52].reshape((self.n_outputs, ))

        z1 = np.dot(X, W1) + b1
        a1 = np.tanh(z1)
        z2 = np.dot(a1, W2) + b2
        a2 = self.af.softmax(z2)
        y_pred = np.argmax(a2, axis=1)
        #y_pred = np.argmax(z2, axis=1)
        return y_pred
    
    def log(self, s):
        best_value = [p.best_value for p in s.particles()]
        best_value_avg = np.mean(best_value)
        best_value_std = np.std(best_value)
        self.metric_loss.append(s.gbest_value)
        print("{0: >5}  {1: >9.3E}  {2: >9.3E}  {3: >9.3E}".format(s.iters, s.gbest_value, best_value_avg, best_value_std))

    def train(self, X, y, beta):
        self.X = X[:400]
        self.y = y[:400]

        #initalize parameters for qdpso 
        NParticle = 25
        MaxIters = 1000
        self.d = (self.n_inputs * self.n_hidden) + (self.n_hidden * self.n_outputs) + self.n_hidden + self.n_outputs
        bounds = [(-1, 1) for i in range(0, self.d)]
        g = beta #1.7 #.96

        s = QDPSO(self.f, NParticle, self.d, bounds, MaxIters, g)
        s.update(callback=self.log, interval=1)

        y_pred = self.predict(X[400:], s.gbest)

        return y_pred