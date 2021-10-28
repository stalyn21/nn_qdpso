import numpy as np
from sklearn.datasets import load_boston

X, y = load_boston(return_X_y=True)
y = np.array(y).reshape((len(y),1))

print(X.shape, y.shape)
#X = X / np.amax(X, axis=0)
#print(X[:10])
y = y/np.amax(y)
print(y[:10])