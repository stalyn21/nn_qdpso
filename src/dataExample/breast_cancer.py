# Import modules
import numpy as np
from sklearn.datasets import load_breast_cancer

# Load the iris dataset
data = load_breast_cancer()

# Store the features as X and the labels as y
X = data.data
y = data.target
y = np.array(y).reshape((len(y),1))

print(np.amax(X))

print("Input: ",X.shape)
print("Output: ",y.shape)