import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles
from neuralNetwork.deepNeuralNetwork import neuralNetwork

#sklearn metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

def qpso_nn():
    n = 500 #register number
    p = 2 #featurs over our data

    #generating the input X and output y (binary vector)
    X, y = make_circles(n_samples=n, factor=0.5, noise=0.05)
    #y = y[:, np.newaxis]

    plt.scatter(X[y==0,0], X[y==0, 1], c="skyblue")
    plt.scatter(X[y==1,0], X[y==1, 1], c="magenta")
    plt.show()
    beta = 1.16

    nn = neuralNetwork()
    #without training forward
    y_p = np.argmax(nn.forward(X[400:]), axis=1)
    cost_pred_0 = mean_squared_error(y[400:], y_p)
    acc_0 = accuracy_score(y[400:], y_p)
    prec_0 = precision_score(y[400:], y_p, average='weighted')


    #With trainning
    y_pred = nn.train(X, y, beta)
    cost_pred1 = mean_squared_error(y[400:], y_pred)
    acc = accuracy_score(y[400:], y_pred)
    prec = precision_score(y[400:], y_pred, average='weighted')

    print("Result without trainning: ")
    print('Prediction/Accuracy/Precision: ',cost_pred_0,'/', acc_0,'/',prec_0)
    print("Result with trainning: ")
    print('Prediction/Accuracy/Precision: ',cost_pred1,'/', acc,'/',prec)
    

    #plotting data
    x_plot = np.arange(1001)
    fig1, ax1 = plt.subplots()
    ax1.plot(x_plot, nn.metric_loss)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Global best value')
    #ax1.set_yscale('log')
    ax1.set_title("The evolution of the gbest value during the trainning with QPSO")
    plt.savefig(f"g_best:{beta}:{cost_pred1}.png")
    plt.show()

def main():
    qpso_nn()

if __name__ == "__main__":
    main()