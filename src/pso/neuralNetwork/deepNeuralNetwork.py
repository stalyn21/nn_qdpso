import numpy as np
import pyswarms as ps

class activationFunction(object):
    def __init__(self):
        self.n_sample_af = 150

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
        self.n_sample_lf = 150

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
        self.n_inputs = 4
        self.n_hidden = 20
        self.n_outputs = 3
        self.n_sample = 150
        self.af = activationFunction()
        self.lf = lossFunction()
        self.metric_loss = []

        # Weight and Bias parameters
        self.W1 = np.random.randn(self.n_inputs, self.n_hidden)
        self.W2 = np.random.randn(self.n_hidden, self.n_outputs)
        self.b1 = np.random.randn(self.n_hidden, )
        self.b2 = np.random.randn(self.n_outputs, )
    
    def forward(self, X, p):
        W1 = p[0:80].reshape((self.n_inputs, self.n_hidden))
        b1 = p[80:100].reshape((self.n_hidden, ))
        W2 = p[100:160].reshape((self.n_hidden, self.n_outputs))
        b2 = p[160:163].reshape((self.n_outputs, ))

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
        n_particles = x.shape[0]
        j = [self.backware(x[i]) for i in range(n_particles)]
        self.metric_loss.append(j)
        return j

    def predict(self, X, p_best):
        W1 = p_best[0:80].reshape((self.n_inputs, self.n_hidden))
        b1 = p_best[80:100].reshape((self.n_hidden, ))
        W2 = p_best[100:160].reshape((self.n_hidden, self.n_outputs))
        b2 = p_best[160:163].reshape((self.n_outputs, ))

        z1 = np.dot(X, W1) + b1
        a1 = np.tanh(z1)
        z2 = np.dot(a1, W2) + b2

        y_pred = np.argmax(z2, axis=1)

        return y_pred

    def train(self, X, y):
        self.X = X
        self.y = y

        # Set-up hyperparameters
        opt = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
        # Create bounds
        """max_bound = 1 * np.ones(9)
        min_bound = - max_bound
        bounds = (min_bound, max_bound)"""
        # Call instance of PSO
        d = (self.n_inputs * self.n_hidden) + (self.n_hidden * self.n_outputs) + self.n_hidden + self.n_outputs
        optimizer = ps.single.GlobalBestPSO(n_particles=25, dimensions=d, options=opt) #bounds=bounds
        # Perform optimization
        best_cost, best_pos = optimizer.optimize(self.f, iters=1000)

        y_pred = self.predict(X, best_pos)

        return y_pred