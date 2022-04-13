#---------------
# import modules
#---------------

import numpy as np
import cv2
import matplotlib.pyplot as plt

from os import listdir
from qpso.qpso import QDPSO
from sklearn.metrics import mean_squared_error
from matplotlib import image #loading img
from skimage import color #gray-scale
#from skimage.transform import resize, rescale

from neuralNetwork.perceptron import perceptron 

#----------------
#Global variables
#----------------

root_path = './dataset/'
train_path = 'seg_train/'
#test_path = 'seg_test/seg_test/'
#pred_path = 'seg_pred/seg_pred/'

# Dataset classes: buildings, forest, glacier, mountain, sea, street
data_clases = np.array(['buildings', 
						'forest', 
						'glacier', 
						'mountain', 
						'sea', 
						'street'])
IMG_HEIGHT = 30
IMG_WIDTH = 30

#---------------------------------
# prepare training and testing set
#---------------------------------

X_train_aux = []
y_train = []

for c in range(len(data_clases)):
	for filename in listdir(root_path + train_path + data_clases[c] + '/'):
		# load image
		img_data = image.imread(root_path + train_path + data_clases[c] + '/' + filename)
		img_data = color.rgb2gray(img_data)
		img_data = cv2.resize(img_data, (IMG_HEIGHT,IMG_WIDTH), interpolation=cv2.INTER_CUBIC)
		#img_data = np.resize(img_data, (IMG_HEIGHT,IMG_WIDTH))
		# store loaded image
		X_train_aux.append(img_data)
		y_train.append(c)
		print('> loaded %s %s' % (filename, img_data.shape))

# convert to array (float 64)
X_train_aux = np.array(X_train_aux, dtype=float)
y_train = np.array(y_train)

# normalize data
#X_train_aux = X_train_aux / 255

print('X_train shape: ', X_train_aux.shape)
print('y_train shape: ', y_train.shape)

# reshape X_train for qpso algorithm

"""X_train = np.reshape(X_train_aux[0], (1, -1))

for i in range(1, len(X_train_aux)):
    X_train_reshape = np.reshape(X_train_aux[i], (1, -1))
    X_train = np.concatenate((X_train, X_train_reshape))
    print('reshape img: ', i)"""

X_train = X_train_aux.reshape(X_train_aux.shape[-3], np.prod(X_train_aux.shape[-2:]))
print('Reshape X_train for QDPSO: ', X_train.shape)

#print('check values of the input: ', X_train[1])
#print('reshape X_train input: ', X_train.shape)

# qdpso neural network algorithm

#----------------
# Load perceptron
#----------------
X_sample = len(X_train) 
X_input = IMG_HEIGHT * IMG_WIDTH
X_class = len(data_clases)

nn = perceptron(X_sample, X_input, X_class)

#---------------------------------------
# variable initialize for qpso algorithm
#---------------------------------------

beta = 1.13 #1.7
n_particulas = 25
max_iter = 1000
gBest = nn.train(X_train, y_train, beta, n_particulas, max_iter)
print('best value', gBest)

# Save global best value
# The file name is 'gBest/bValue_beta_nParticles_maxIter_imgHEIGHTximgWIDTH_rgbChannel.npy'
np.save("gBest_113_25_1000_30x30_1.npy", gBest)
# Save best value during the trainning 
np.save("bValue_113_25_1000_30x30_1.npy", nn.metric_loss)

y_pred = np.argmax(nn.forward(X_train, gBest), axis=1)
cost_pred1 = mean_squared_error(y_train, y_pred)
print ('prediction cost with training',cost_pred1)
