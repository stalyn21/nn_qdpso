#---------------
# import modules
#---------------

import numpy as np
import cv2
import mahotas

from os import listdir
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import MinMaxScaler
# Uncomment to feature reduction
#from sklearn.manifold import Isomap
from sklearn.model_selection import train_test_split
from collections import Counter


from neuralNetwork.perceptron_qpso import perceptron 

#----------------
#Global variables
#----------------
root_path = './dataset2/'

# Dataset classes: 
data_clases = np.array(['cloudy', 
						'rain', 
						'shine', 
						'sunrise'])

n_particulas = 100
max_iter = 1000
n_training = 10

IMG_HEIGHT = 150
IMG_WIDTH = 150
# uncomment to reduction feature
#dimension = 14

#---------------------------------
# Extract Global features
#---------------------------------
bins = 4 

# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick

# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()

#---------------------------------
# Prepare training and testing set
#---------------------------------
X_train_aux = []
y_train_aux = []

for c in range(len(data_clases)):
	for filename in listdir(root_path + data_clases[c] + '/'):
		# load image
		img_data = cv2.imread(root_path + data_clases[c] + '/' + filename) 
		img_data = cv2.resize(img_data, (IMG_HEIGHT, IMG_WIDTH))
        # Global Feature extraction
		fv_hu_moments = fd_hu_moments(img_data)
		#print(fv_hu_moments.shape)
		fv_haralick = fd_haralick(img_data)
		#print(fv_haralick.shape)        
		fv_histogram  = fd_histogram(img_data)
		#print(fv_histogram.shape)
		# Concatenate global features
		global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
		X_train_aux.append(global_feature)
		y_train_aux.append(c)
		print('> loaded %s %s' % (filename, img_data.shape))
		print('> global features %s %s' % (filename, global_feature.shape))

# convert to array
X_train_aux = np.array(X_train_aux)
y_train_aux = np.array(y_train_aux)

print('X_train shape: ', X_train_aux.shape)
print('y_train shape: ', y_train_aux.shape)

#---------------------------------
# Normalizind dataset
#---------------------------------
t = MinMaxScaler()
t.fit(X_train_aux)
X_train_aux = t.transform(X_train_aux)

#---------------------------------
# Feature reduction
#---------------------------------
# Uncomment to reduction feature
#embedding = Isomap(n_components=dimension, n_neighbors=100)
#X_transformed = embedding.fit_transform(X_train_aux)
#print ('Shape of X_train transformed: ', X_transformed.shape)

#---------------------------------
# Split dataset
#---------------------------------
# if uncomment feature reduction reempleces X_train_aux to X_transformed 
X_train_bal, X_test_bal, y_train_bal, y_test_bal = train_test_split(X_train_aux, y_train_aux, test_size=0.3, random_state=100)
print(f"Training target statistics: {Counter(y_train_bal)}")
print(f"Testing target statistics: {Counter(y_test_bal)}")

X_sample = len(X_train_bal)
X_input = len(X_train_bal[1])
X_class = len(np.unique(y_train_bal))

X_train = X_train_bal
y_train = y_train_bal
X_test = X_test_bal
y_test = y_test_bal

#-----------------------------------------
# Variable initialize for qdpso algorithm
#-----------------------------------------
beta = 1.13 #1.13

gBest_value = []
gBest = []
cost_test = []
metric_train = []

for i in range(n_training):
    # load perceptron
    nn = perceptron(X_sample, X_input, X_class)
    gBest.append( nn.train(X_train, y_train, beta, n_particulas, max_iter) )
    gBest_value.append(nn.g_best_value)
    metric_train.append(nn.g_best_value_iter)
    y_test_pred = np.argmax(nn.forward(X_test, gBest[i]), axis=1)
    cost_test.append(mean_squared_error(y_test, y_test_pred))
    print ('Test prediction cost with training: ', cost_test[i]) 
    min_cost = min(cost_test)
    print ("Min test prediction cost: ", min_cost)
    if  cost_test[i] <= min_cost:       
        np.save("mcw_gBest_113_100_1000_84_qdpso.npy", gBest[i])
        np.save("mcw_gBestIter_113_100_1000_84_qdpso.npy", nn.g_best_value_iter)
        np.save("mcw_avgBest_113_100_1000_84_qdpso.npy", nn.avg_best_value)
        np.save("mcw_stdBest_113_100_1000_84_qdpso.npy", nn.std_best_value)
        
print("=====================================================================")
print("=====================================================================")
print("Save train metric ....")
np.save("mcw_metric_113_100_1000_84_qdpso.npy", metric_train)
print("The best training is in iteration: ", cost_test.index(min(cost_test)))
print("The golbal best value is: ", gBest_value[cost_test.index(min(cost_test))])
print("Test prediction cost: ", min(cost_test))
print("=====================================================================")
print("=====================================================================")

#-----------------------------------------
# Training and testing error
#-----------------------------------------
model_load = np.load('mcw_gBest_113_100_1000_84_qdpso.npy')
print("=====================================================================")
print("=====================================================================")
print("Training and Testing error")
y_train_pred_load = np.argmax(nn.forward(X_train, model_load), axis=1)
cost_train_load = mean_squared_error(y_train, y_train_pred_load)
acc_train_load = accuracy_score(y_train, y_train_pred_load)
print('MSE of training phase: ', cost_train_load, ' ACC score of training phase: ', acc_train_load)

y_test_pred_load = np.argmax(nn.forward(X_test, model_load), axis=1)
cost_test_load = mean_squared_error(y_test, y_test_pred_load)
acc_test_load = accuracy_score(y_test, y_test_pred_load)
print('MSE of testing phase: ', cost_test_load, ' ACC score of testing phase: ', acc_test_load)
print("=====================================================================")
print("=====================================================================")
