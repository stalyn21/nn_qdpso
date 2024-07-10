#--------------
#import modules
#--------------

import numpy as np
import threading
from timeit import default_timer

from sklearn.metrics import mean_squared_error
from qpso.qpso import QDPSO
from multiprocessing import Manager

#-------------------------
#Activation function class
#-------------------------

class activationFunction(object):
    def __init__(self, X_sample):
        self.n_sample_af = X_sample

    def softmax(self, z):
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

    def mse(self, y, y_hat):
        return np.sum(0.5*(y - y_hat)**2)

#-----------------
# Perceptron Class
#-----------------

class perceptron(object):
    def __init__(self, X_sample, X_input, X_class):
        
        self.condition = threading.Condition()

        # initialize parameters
        self.n_inputs = X_input
        self.n_hidden = X_input * 3
        self.n_outputs = X_class
        self.n_sample = X_sample

        manager = Manager()
        self.timings = manager.list()
        self.shared_dict = manager.dict()
        
        self.af = activationFunction(X_sample)
        self.lf = lossFunction(X_sample)
        self.g_best_value_iter_hidden = []
        self.avg_best_value_hidden = []
        self.std_best_value_hidden = []

        self.g_best_value_iter_output = []
        self.avg_best_value_output = []
        self.std_best_value_output = []

        self.d = (self.n_inputs * self.n_hidden) + (self.n_hidden * self.n_outputs) + self.n_hidden + self.n_outputs
        self.d_hidden = (self.n_inputs * self.n_hidden) + self.n_hidden
        self.d_output = (self.n_hidden * self.n_outputs) + self.n_outputs

        # Weight and Bias parameters
        self.W1 = np.random.randn(self.n_inputs, self.n_hidden)
        self.W2 = np.random.randn(self.n_hidden, self.n_outputs)
        self.b1 = np.random.randn(self.n_hidden, )
        self.b2 = np.random.randn(self.n_outputs, )

        # parameters random initialization
        self.pam = np.random.randn(self.d, )
    
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
    
    def forward_train(self, X, p, h):
        if (h==True):
            p = np.array(p).reshape((self.d_hidden, 1))
            W1_lim = self.n_inputs * self.n_hidden
            W1 = p[0:W1_lim].reshape((self.n_inputs, self.n_hidden))
            b1_lim = W1_lim + self.n_hidden
            b1 = p[W1_lim:b1_lim].reshape((self.n_hidden, ))
            W2 = self.W2
            b2 = self.b2
            
        else:
            p = np.array(p).reshape((self.d_output, 1))
            W1 = self.W1
            b1 = self.b1
            W2_lim = self.n_hidden * self.n_outputs
            W2 = p[0:W2_lim].reshape((self.n_hidden, self.n_outputs))
            b2_lim = W2_lim + self.n_outputs
            b2 = p[W2_lim:b2_lim].reshape((self.n_outputs, ))

        z1 = np.dot(X, W1) + b1
        a1 = np.tanh(z1)
        z2 = np.dot(a1, W2) + b2
        y_hat = self.af.softmax(z2)
        return y_hat
    
    def backware(self, p, h):
        y_hat = self.forward_train(self.X, p, h)
        loss = self.lf.crossEntropy(self.y, y_hat)
        return loss

    def f_hidden(self, x):
        j = self.backware(x, True)
        return j
    
    def f_output(self, x):
        j = self.backware(x, False)
        return j
    
    def log_hidden(self, s):
        with self.cond:
            while not self.start_hidden:
                self.cond.wait()
            print("> Training hidden layer")
            best_value = [p.best_value for p in s.particles()]
            best_value_avg = np.mean(best_value)
            best_value_std = np.std(best_value)
            self.g_best_value_iter_hidden.append(s.gbest_value)
            self.avg_best_value_hidden.append(best_value_avg)
            self.std_best_value_hidden.append(best_value_std)
            # Update parameters of hidden layer
            y_hat_hidden = np.argmax(self.forward_train(self.X_test, s.gbest, True), axis=1)
            cost_test_hidden = mean_squared_error(self.y_test, y_hat_hidden)
            if (cost_test_hidden < self.hidden_error):
                self.hidden_error = cost_test_hidden
                self.W1 = s.gbest[0:self.n_inputs * self.n_hidden].reshape((self.n_inputs, self.n_hidden))
                self.b1 = s.gbest[self.n_inputs * self.n_hidden:(self.n_inputs * self.n_hidden) + self.n_hidden].reshape((self.n_hidden, ))
                self.pam[:self.d_hidden] = s.gbest
            print("{0: >5}  {1: >9.3E}  {2: >9.3E}  {3: >9.3E}".format(s.iters, s.gbest_value, best_value_avg, best_value_std))
            print("Current error: ", cost_test_hidden)
            print("Best error: ", self.hidden_error)
            self.start_hidden = False
            self.start_output = True
            self.shared_dict['pam'] = self.pam
            self.shared_dict['g_best_value_iter_hidden'] = self.g_best_value_iter_hidden
            self.shared_dict['avg_best_value_hidden'] = self.avg_best_value_hidden
            self.shared_dict['std_best_value_hidden'] = self.std_best_value_hidden
            self.cond.notify_all()

    def log_output(self, s):
        with self.cond:
            while not self.start_output:
                self.cond.wait()
            print("> Training output layer")
            best_value = [p.best_value for p in s.particles()]
            best_value_avg = np.mean(best_value)
            best_value_std = np.std(best_value)
            self.g_best_value_iter_output.append(s.gbest_value)
            self.avg_best_value_output.append(best_value_avg)
            self.std_best_value_output.append(best_value_std)
            # Update parameters of output layer
            y_hat_output = np.argmax(self.forward_train(self.X_test, s.gbest, False), axis=1)
            cost_test_output = mean_squared_error(self.y_test, y_hat_output)
            if (cost_test_output < self.output_error):
                self.output_error = cost_test_output
                self.W2 = s.gbest[0:self.n_hidden * self.n_outputs].reshape((self.n_hidden, self.n_outputs))
                self.b2 = s.gbest[self.n_hidden * self.n_outputs:(self.n_hidden * self.n_outputs) + self.n_outputs].reshape((self.n_outputs, ))
                self.pam[self.d_hidden:] = s.gbest
            print("{0: >5}  {1: >9.3E}  {2: >9.3E}  {3: >9.3E}".format(s.iters, s.gbest_value, best_value_avg, best_value_std))
            self.start_output = False
            self.start_hidden = True
            self.shared_dict['pam'] = self.pam
            self.shared_dict['g_best_value_iter_output'] = self.g_best_value_iter_output
            self.shared_dict['avg_best_value_output'] = self.avg_best_value_output
            self.shared_dict['std_best_value_output'] = self.std_best_value_output
            self.cond.notify_all()

    def one_hot_encode(self, Y):
        num_unique = len(np.unique(np.array(Y)))
        zeros = np.zeros((len(Y), num_unique))
        zeros[range(len(Y)), Y] = 1
        return zeros

    def train(self, X, y, X_test, y_test, beta, n_particulas, max_iter, hidden_error, output_error, *parameters):

        self.g_best_value_iter_hidden = []
        self.avg_best_value_hidden = []
        self.std_best_value_hidden = []

        self.g_best_value_iter_output = []
        self.avg_best_value_output = []
        self.std_best_value_output = []

        print(">> Training nn with QDPSO")
        self.X = X
        self.y = self.one_hot_encode(y)
        self.X_test = X_test
        self.y_test = y_test
        self.hidden_error = hidden_error
        self.output_error = output_error


        if len(parameters)!=0:
            self.parameters = np.array(parameters).reshape(self.d,)
            W1_lim = self.n_inputs * self.n_hidden
            b1_lim = W1_lim + self.n_hidden
            W2_lim = b1_lim + (self.n_hidden * self.n_outputs)
            b2_lim = W2_lim + self.n_outputs
            # Update parameters for training
            self.W1 = self.parameters[0:W1_lim].reshape((self.n_inputs, self.n_hidden))
            self.b1 = self.parameters[W1_lim:b1_lim].reshape((self.n_hidden, ))            
            self.W2 = self.parameters[b1_lim:W2_lim].reshape((self.n_hidden, self.n_outputs))            
            self.b2 = self.parameters[W2_lim:b2_lim].reshape((self.n_outputs, ))


        
        #initalize parameters for qdpso 
        self.NParticle = n_particulas
        self.MaxIters = max_iter
        self.bounds_hidden = [(-1, 1) for _ in range(0, self.d_hidden)]
        self.bounds_output = [(-1, 1) for _ in range(0, self.d_output)]
        self.g = beta #1.7 #.96
        
        s_hidden = QDPSO(self.f_hidden, self.NParticle, self.d_hidden, self.bounds_hidden, self.MaxIters, self.g)
        s_output = QDPSO(self.f_output, self.NParticle, self.d_output, self.bounds_output, self.MaxIters, self.g)

        self.cond = threading.Condition()
        self.start_hidden = False
        self.start_output = True

        #thread1 = threading.Thread(target=self.s_hidden.update(callback=self.log_hidden, interval=1))
        #thread2 = threading.Thread(target=self.s_output.update(callback=self.log_output, interval=1))

        thread1 = threading.Thread(target=s_hidden.update, args=(self.log_hidden, 1))
        thread2 = threading.Thread(target=s_output.update, args=(self.log_output, 1))

        thread1.start()
        thread2.start()
        total_start_time_hidden = default_timer()  
        thread1.join()
        total_end_time_hidden = default_timer()

        total_start_time_output = default_timer()
        thread2.join()
        total_end_time_output = default_timer()

        self.param = self.shared_dict['pam']
        self.g_best_value_iter_hidden = self.shared_dict['g_best_value_iter_hidden']
        self.avg_best_value_hidden = self.shared_dict['avg_best_value_hidden']
        self.std_best_value_hidden = self.shared_dict['std_best_value_hidden']
        self.g_best_value_iter_output = self.shared_dict['g_best_value_iter_output']
        self.avg_best_value_output = self.shared_dict['avg_best_value_output']
        self.std_best_value_output = self.shared_dict['std_best_value_output']

         # Añadir las duraciones totales a la lista de tiempos
        self.timings.append(("total_hidden", total_start_time_hidden, total_end_time_hidden))
        self.timings.append(("total_output", total_start_time_output, total_end_time_output))

        self.g_best_value = np.array([s_hidden.gbest_value, s_output.gbest_value])
        self.g_best_value_iter = np.array([self.g_best_value_iter_hidden, self.g_best_value_iter_output])
        self.avg_best_value = np.array([self.avg_best_value_hidden, self.avg_best_value_output])
        self.std_best_value = np.array([self.std_best_value_hidden, self.std_best_value_output])

        return self.param