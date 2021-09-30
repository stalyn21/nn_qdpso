import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from neuralNetwork.deepNeuralNetwork import neuralNetwork

from sklearn.metrics import mean_squared_error


# load the data
data = load_iris()
X = data.data
y = data.target
#y = np.array(y).reshape((len(y),1))

nn = neuralNetwork()
#loss1 = nn.train(X, y)
#print(loss1)
#cost_pred = 1.0 - (nn.train(X, y) == y).mean()
y_pred = nn.train(X, y)
cost_pred1 = mean_squared_error(y, y_pred)

print('value prediction: ', cost_pred1)
#print('shape of plotting: ', np.array(nn.metric_loss).shape)

metric = (np.array(nn.metric_loss)[:25000]).reshape((1000, 25))
print(metric.shape)


#plotting
x_plot = np.arange(1000)
y_plot = []

for i in range(1000):
    y_plot.append(metric[i][0])

fig1, ax1 = plt.subplots()
ax1.plot(x_plot, y_plot)
ax1.set_xlabel('epoch')
ax1.set_ylabel('loss')
ax1.set_title("The evolution of the loss during the trainning with QPSO 1.7-2")
plt.savefig("loss2.png")
plt.show()