import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, make_circles
from neuralNetwork.deepNeuralNetwork import neuralNetwork

#sklearn metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

n = 500 #register number
p = 2 #featurs over our data

#generating the input X and output y (binary vector)
X, y = make_circles(n_samples=n, factor=0.5, noise=0.05)
#y = y[:, np.newaxis]

plt.scatter(X[y==0,0], X[y==0, 1], c="skyblue")
plt.scatter(X[y==1,0], X[y==1, 1], c="magenta")
plt.show()
#------------------Iris Data ----------
# load the data
#data = load_iris()
#X = data.data
#y = data.target
#y = np.array(y).reshape((len(y),1))
#---------------------------------------
beta = 1.16

nn = neuralNetwork()
#loss1 = nn.train(X, y)
#print(loss1)
#cost_pred = 1.0 - (nn.train(X, y) == y).mean()
y_pred = nn.train(X, y, beta)
cost_pred1 = mean_squared_error(y[400:], y_pred)
acc = accuracy_score(y[400:], y_pred)
prec = precision_score(y[400:], y_pred, average='weighted')

print('Prediction/Accuracy/Precision: ',cost_pred1,'/', acc,'/',prec)
#print('shape of plotting: ', np.array(nn.metric_loss).shape)

#metric = (np.array(nn.metric_loss)[:25000]).reshape((1000, 25))
#print(len(nn.metric_loss))


#plotting
x_plot = np.arange(1001)
#y_plot = []

#for i in range(1000):
    #y_plot.append(metric[i][0])

fig1, ax1 = plt.subplots()
ax1.plot(x_plot, nn.metric_loss)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Global best value')
#ax1.set_yscale('log')
ax1.set_title("The evolution of the gbest value during the trainning with QPSO")
plt.savefig(f"g_best:{beta}:{cost_pred1}.png")
plt.show()