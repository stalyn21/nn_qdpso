import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from neuralNetwork.deepNeuralNetwork import neuralNetwork

#sklearn metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

def qpso_nn():
    # load the data
    data = load_iris()
    X = data.data
    y = data.target
    #y = np.array(y).reshape((len(y),1))

    beta = 1.13

    nn = neuralNetwork()

    #without trainning (Forware)
    y_p = np.argmax(nn.forward(X), axis=1)
    cost_pred_0 = mean_squared_error(y, y_p)
    acc_0 = accuracy_score(y, y_p)
    prec_0 = precision_score(y, y_p, average='weighted')

    #with trainning
    y_pred = nn.train(X, y, beta)
    cost_pred1 = mean_squared_error(y, y_pred)
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, average='weighted')

    print("Result without trainning: ")
    print('Prediction/Accuracy/Precision: ',cost_pred_0,'/', acc_0,'/',prec_0)
    print("Result with trainning: ")
    print('Prediction/Accuracy/Precision: ',cost_pred1,'/', acc,'/',prec)

    #plotting
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