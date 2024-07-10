# Training ANN with QPSO using One Swarm

## Requirements
- GNU/Linux (Kali Linux in this case)
- Python 3.9.2
- Pyswarms
- QDPSO
- Scikit-learn 
- Numpy
- Matplotlip
- Opencv-python (CV2)
- Joblib
- Pickle
- Seaborn
- Mahotas

To install the requirements execute the [req](./req.txt) file or execute the following command:
```
$ pip install -r req.txt
```
Installing the scikit-learn package with this command:
```
$ pip install -U scikit-learn==1.0
```
## Structure
This section has benchmark, feature reduction, increasing classes and samples, and images classification results. Moreover, it has the heatmaps of bechmark and images classification. Finally, we use the circle, iris, wine, breast canser, and Multi-class Waether datasets for this project. 

## Script usage for image classification
To run the algorithm use the [Multi-class Weather dataset](https://www.kaggle.com/datasets/somesh24/multiclass-images-for-weather-classification), the [main.py](./main.py) file and execute the following command: 
```
$ python3 main.py
```

<b> Note: </b> If you have the following output during the execution, use the following command:
```
OUTPUT:
Traceback (most recent call last):
  File "/home/nn_qdpso-main/src/main.py", line 6, in <module>
    import cv2
  File "/usr/local/lib/python3.10/dist-packages/cv2/__init__.py", line 8, in <module>
    from .cv2 import *
ImportError: libGL.so.1: cannot open shared object file: No such file or directory

COMMAND
$ sudo apt install -y python3-opencv
```

In the [main](./main.py) file, you can change the images size, alfa or beta value, the particles number, the max iterations and another parameters, see below.
```
# At the beginnig in global variables section
n_particulas = 100
max_iter = 1000
n_training = 10

IMG_HEIGHT = 150
IMG_WIDTH = 150
# uncomment to reduction feature
#dimension = 14

# At the middle in feature reduction section
# Uncomment to reduction feature
#embedding = Isomap(n_components=dimension, n_neighbors=100)
#X_transformed = embedding.fit_transform(X_train_aux)
#print ('Shape of X_train transformed: ', X_transformed.shape)

# At the ending in variables for qdpso section
beta = 1.13
```

<b>Note:</b> If you have a problem loading the dataset, remove the rain141 and shine 131 images or resize the image pixels to 128x128.
