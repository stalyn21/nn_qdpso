import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles

n = 500 #register number
p = 2 #featurs over our data

#generating the input X and output y (binary vector)
X, y = make_circles(n_samples=n, factor=0.5, noise=0.05)
y = y[:, np.newaxis]

print(X.shape)
print(y.shape)

plt.scatter(X[y[:, 0]==0,0], X[y[:,0]==0, 1], c="skyblue")
plt.scatter(X[y[:, 0]==1,0], X[y[:,0]==1, 1], c="magenta")
plt.show()