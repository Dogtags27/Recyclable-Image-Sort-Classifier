
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
model = keras.models.load_model("_model.h5")
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import skimage.io
import tensorflow 
import tqdm
import glob

from tqdm import tqdm 

from skimage.io import imread, imshow
from skimage.transform import resize

# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications.vgg16 import VGG16
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import InputLayer, Dense, Flatten, BatchNormalization, Dropout, Activation
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# %matplotlib inline
imgPath='DATASET\TEST\O\O_12838.jpg'

from PIL import Image
from numpy import asarray
 
 
# load the image and convert into
# numpy array
img = load_img(imgPath, target_size=(224,224))
img = img_to_array(img)
img = img / 255
imshow(img)
plt.axis('off')

img = np.expand_dims(img,axis=0)
# answer = model.predict_proba(img)
predict_prob=model.predict(img)

predict_classes=np.argmax(predict_prob,axis=1)
print(predict_classes[0])
if predict_classes > 0.5:
    print("The image belongs to Recycle waste category")
else:
    print("The image belongs to Organic waste category ")

# test_datagen  = ImageDataGenerator(rescale = 1.0 / 255.0)

# numpydata = asarray(img)
# nnumpydata = np.resize(numpydata, (224,224,3))
# nnumpydata = np.resize(numpydata, [1,224,224,3])
# test_data = test_datagen.flow(nnumpydata,batch_size = 1)

# asarray() class is used to convert
# PIL images into NumPy arrays

 
# # <class 'numpy.ndarray'>
# print(type(numpydata))
 
# #  shape
# print(numpydata.shape)

# nnumpydata = np.resize(numpydata, (224,224,3))
# nnumpydata = np.resize(numpydata, [1,224,224,3])
# print(nnumpydata.shape)
# # k=(model.predict(nnumpydata))
# k=(model.predict(test_data))
# print(k)
# # print(np.round_(k))
# # k=k[0][0]
# # print(k)
# # s = "Object is recyclable" if k < 0.1 else "Object is not recyclable"
# # print(k)
print("done")