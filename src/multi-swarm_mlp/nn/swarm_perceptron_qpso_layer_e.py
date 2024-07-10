#--------------
#import modules
#--------------

import numpy as np
from timeit import default_timer

from sklearn.metrics import mean_squared_error
from qpso.qpso import QDPSO
from multiprocessing import Pool

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
        # initialize parameters
        np.random.seed(444)
        self.n_inputs = X_input
        self.n_hidden = X_input * 3
        self.n_outputs = X_class
        self.n_sample = X_sample
        self.timings = []  
                
        self.af = activationFunction(X_sample)
        self.lf = lossFunction(X_sample)        

        self.d = (self.n_inputs * self.n_hidden) + (self.n_hidden * self.n_outputs) + self.n_hidden + self.n_outputs
        self.d_hidden = (self.n_inputs * self.n_hidden) + self.n_hidden
        self.d_output = (self.n_hidden * self.n_outputs) + self.n_outputs

        # parameters random initialization
        self.pam = np.random.randn(self.d, )
        self.error_hidden = 0
        self.error_output = 0

        # Weight and Bias parameters
        W1_lim = self.n_inputs * self.n_hidden
        self.W1 = self.pam[0:W1_lim].reshape((self.n_inputs, self.n_hidden))
        b1_lim = W1_lim + self.n_hidden
        self.b1 = self.pam[W1_lim:b1_lim].reshape((self.n_hidden, ))
        W2_lim = b1_lim + (self.n_hidden * self.n_outputs)
        self.W2 = self.pam[b1_lim:W2_lim].reshape((self.n_hidden, self.n_outputs))
        b2_lim = W2_lim + self.n_outputs
        self.b2 = self.pam[W2_lim:b2_lim].reshape((self.n_outputs, ))
    
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
    
    def forward_train(self, X, p, h, *params):
        if (h==True and len(params)==0):
            p = np.array(p).reshape((self.d_hidden, 1))
            W1_lim = self.n_inputs * self.n_hidden
            W1 = p[0:W1_lim].reshape((self.n_inputs, self.n_hidden))
            b1_lim = W1_lim + self.n_hidden
            b1 = p[W1_lim:b1_lim].reshape((self.n_hidden, ))
            W2 = self.W2
            b2 = self.b2
            
        elif (h==False and len(params)==0):
            p = np.array(p).reshape((self.d_output, 1))
            W1 = self.W1
            b1 = self.b1
            W2_lim = self.n_hidden * self.n_outputs
            W2 = p[0:W2_lim].reshape((self.n_hidden, self.n_outputs))
            b2_lim = W2_lim + self.n_outputs
            b2 = p[W2_lim:b2_lim].reshape((self.n_outputs, ))

        elif (h==True and len(params)!=0):
            p = np.array(p).reshape((self.d_hidden, 1))
            W1_lim = self.n_inputs * self.n_hidden
            W1 = p[0:W1_lim].reshape((self.n_inputs, self.n_hidden))
            b1_lim = W1_lim + self.n_hidden
            b1 = p[W1_lim:b1_lim].reshape((self.n_hidden, ))
            params_aux = np.array(params[0][self.d_hidden:]).reshape((self.d_output, 1))
            W2_lim = self.n_hidden * self.n_outputs
            W2 = params_aux[0:W2_lim].reshape((self.n_hidden, self.n_outputs))
            b2_lim = W2_lim + self.n_outputs
            b2 = params_aux[W2_lim:b2_lim].reshape((self.n_outputs, ))
        else:
            params_aux = np.array(params[0][:self.d_hidden]).reshape((self.d_hidden, 1))
            W1_lim = self.n_inputs * self.n_hidden
            W1 = params_aux[0:W1_lim].reshape((self.n_inputs, self.n_hidden))
            b1_lim = W1_lim + self.n_hidden
            b1 = params_aux[W1_lim:b1_lim].reshape((self.n_hidden, ))
            p = np.array(p).reshape((self.d_output, 1))
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
        if self.parameters is not None and (isinstance(self.parameters, np.ndarray) and self.parameters.size > 0):
            y_hat = self.forward_train(self.X, p, h, self.parameters)
        else:
            y_hat = self.forward_train(self.X, p, h)

        loss = self.lf.crossEntropy(self.y, y_hat)
        return loss

    def f_hidden(self, x):
        j = self.backware(x, True)
        return j
    
    def f_output(self, x):
        j = self.backware(x, False)
        return j
    
    def log_hidden(self, s, data_hidden):
        best_value = [p.best_value for p in s.particles()]
        best_value_avg = np.mean(best_value)
        best_value_std = np.std(best_value)
        y_hat_pred_hidden =  np.argmax(self.forward_train(self.X_test, s.gbest, True), axis=1)
        cost_pred_hidden = mean_squared_error(self.y_test, y_hat_pred_hidden)
        # Update the best parameters of the hidden layer
        if (cost_pred_hidden < self.error_hidden):
            self.error_hidden = cost_pred_hidden
            data_hidden['param_h'] = s.gbest
        self.g_best_value_iter_hidden.append(s.gbest_value)
        self.avg_best_value_hidden.append(best_value_avg)
        self.std_best_value_hidden.append(best_value_std)
        data_hidden['g_best_value_iter_hidden'] = self.g_best_value_iter_hidden
        data_hidden['avg_best_value_hidden'] = self.avg_best_value_hidden
        data_hidden['std_best_value_hidden'] = self.std_best_value_hidden
        data_hidden['error_hidden'] = self.error_hidden
        print("{0: >5}  {1: >9.3E}  {2: >9.3E}  {3: >9.3E}".format(s.iters, s.gbest_value, best_value_avg, best_value_std))
        print("Current hidden error: ", cost_pred_hidden)
        print("Best hidden error: ", self.error_hidden)        

    def log_output(self, s, data_output):
        best_value = [p.best_value for p in s.particles()]
        best_value_avg = np.mean(best_value)
        best_value_std = np.std(best_value)
        y_hat_pred_output = np.argmax(self.forward_train(self.X_test, s.gbest, False), axis=1)
        cost_pred_output = mean_squared_error(self.y_test, y_hat_pred_output)
        # Update the best parameters of the output layer
        if (cost_pred_output < self.error_output):
            self.error_output = cost_pred_output
            data_output['param_o'] = s.gbest
        self.g_best_value_iter_output.append(s.gbest_value)
        self.avg_best_value_output.append(best_value_avg)
        self.std_best_value_output.append(best_value_std)
        data_output['g_best_value_iter_output'] = self.g_best_value_iter_output
        data_output['avg_best_value_output'] = self.avg_best_value_output
        data_output['std_best_value_output'] = self.std_best_value_output
        data_output['error_output'] = self.error_output
        print("{0: >5}  {1: >9.3E}  {2: >9.3E}  {3: >9.3E}".format(s.iters, s.gbest_value, best_value_avg, best_value_std))
        print("Current output error: ", cost_pred_output)
        print("Best output error: ", self.error_output) 
        
    def one_hot_encode(self, Y):
        num_unique = len(np.unique(np.array(Y)))
        zeros = np.zeros((len(Y), num_unique))
        zeros[range(len(Y)), Y] = 1
        return zeros
    
    def parallel_update_hidden(self, NParticle, bounds_hidden, MaxIters, g):
        data_hidden = {}
        start_time_hidden = default_timer()
        print("> Training hidden layer")
        s_hidden = QDPSO(self.f_hidden, NParticle, self.d_hidden, bounds_hidden, MaxIters, g)
        s_hidden.update(callback=lambda s: self.log_hidden(s, data_hidden), interval=1)
        end_time_hidden = default_timer()
        time = ("total_hidden", start_time_hidden, end_time_hidden)
        data_hidden['g_best_value'] = s_hidden.gbest_value
        data_hidden['time'] = time
        return data_hidden

    def parallel_update_output(self, NParticle, bounds_output, MaxIters, g):
        data_output = {}
        start_time_output = default_timer()
        print("> Training output layer")
        s_output = QDPSO(self.f_output, NParticle, self.d_output, bounds_output, MaxIters, g)
        s_output.update(callback=lambda s: self.log_output(s, data_output), interval=1)
        end_time_output = default_timer()
        time = ("total_output", start_time_output, end_time_output)
        data_output['g_best_value'] = s_output.gbest_value
        data_output['time'] = time
        return data_output

    def unpack_and_call(self, func, args):
        return func(*args)

    def train(self, X, y, X_test, y_test, beta, n_particulas, max_iter, hidden_error, output_error, i, *parameters):
        self.g_best_value_iter_hidden = []
        self.avg_best_value_hidden = []
        self.std_best_value_hidden = []

        self.g_best_value_iter_output = []
        self.avg_best_value_output = []
        self.std_best_value_output = []

        print(">> Training nn with QDPSO and difrential swarm by layers")
        self.X = X
        self.y = self.one_hot_encode(y)        
        self.error_hidden = hidden_error
        self.error_output = output_error
        self.X_test = X_test
        self.y_test = y_test        

        if parameters is not None and len(parameters) > 0:
            self.parameters = np.array(parameters).reshape(self.d,)
        else:
            self.parameters = parameters

        #initalize parameters for qdpso 
        NParticle = n_particulas
        MaxIters = max_iter
        bounds_hidden = [(-1, 1) for _ in range(0, self.d_hidden)]
        bounds_output = [(-1, 1) for _ in range(0, self.d_output)]
        g = beta #1.7 #.96

        start_time_training = default_timer()
        with Pool(processes=2) as pool:
            resp_output = pool.apply_async(self.parallel_update_output, (NParticle, bounds_output, MaxIters, g))
            resp_hidden = pool.apply_async(self.parallel_update_hidden, (NParticle, bounds_hidden, MaxIters, g))            
            result_output = resp_output.get()
            result_hidden = resp_hidden.get()
        end_time_training = default_timer()

        self.timings.append(result_output['time'])
        self.timings.append(result_hidden['time'])
        self.timings.append((f"total_training_{i}", start_time_training, end_time_training))
        
        if 'param_o' not in result_output:
            self.pam[self.d_hidden:] = self.parameters[self.d_hidden:]
        else:
            self.pam[self.d_hidden:] = result_output['param_o']
        if 'param_h' not in result_hidden:
            self.pam[:self.d_hidden] = self.parameters[:self.d_hidden]
        else:
            self.pam[:self.d_hidden] = result_hidden['param_h']
            
        self.g_best_value_iter_hidden = result_hidden['g_best_value_iter_hidden']
        self.g_best_value_iter_output = result_output['g_best_value_iter_output']
        self.g_best_value = np.array([result_output['g_best_value'], result_hidden['g_best_value']])
        self.g_best_value_iter = np.array([self.g_best_value_iter_hidden, self.g_best_value_iter_output])
        self.avg_best_value = np.array([result_hidden['avg_best_value_hidden'], result_output['avg_best_value_output']])
        self.std_best_value = np.array([result_hidden['std_best_value_hidden'], result_output['std_best_value_output']])
        self.error_hidden = result_hidden['error_hidden']
        self.error_output = result_output['error_output']
        return self.pam