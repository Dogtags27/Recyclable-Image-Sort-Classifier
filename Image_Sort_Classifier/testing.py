# import tensorflow as tf
# from tensorflow import keras
# import numpy as np
# # import cv2
# from keras.preprocessing.image import *
# # from tensorflow.keras import *
# # with open("Image_Sort_Cam_Model.pkl", "rb") as f:
# #     model = pickle.load(f)
# import matplotlib.pyplot as plt

# # model = pickle.load(open("Image_Sort_Cam_Model.pkl", 'rb'))

# model = keras.models.load_model("my_model.h5")
# # model.compile(optimizer=your_optimizer, loss=[loss1, loss2, ...], metrics==[metrick1, metrick2, ...])
# # model.summary()

# from PIL import Image
# # img = Image.open('DATASET/TRAIN/R/R_3.jpg')
# # cv2.imshow(img)
# # img_data = np.asarray(img)
# # plt.show(img)

# imgPath='DATASET/TRAIN/R/R_293.jpg'
# # img = load_img(imgPath)
# # data = img_to_array(img)

# # img_data = img_data.reshape(-1,224,224,3)

# # model.predict(img_data)

# # print('Model Loaded.')

# import cv2
 
# # # To read image from disk, we use
# # # cv2.imread function, in below method,
# # img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
 
# # # Creating GUI window to display an image on screen
# # # first Parameter is windows title (should be in string format)
# # # Second Parameter is image array
# # cv2.imshow("image", img)
 
# # # To hold the window on screen, we use cv2.waitKey method
# # # Once it detected the close input, it will release the control
# # # To the next line
# # # First Parameter is for holding screen for specified milliseconds
# # # It should be positive integer. If 0 pass an parameter, then it will
# # # hold the screen until user close it.
# # cv2.waitKey(0)
 
# # # It is for removing/deleting created GUI window from screen
# # # and memory
# # cv2.destroyAllWindows()

# from PIL import Image
# from numpy import asarray
 
 
# # load the image and convert into
# # numpy array
# img = Image.open(imgPath)
 
# # asarray() class is used to convert
# # PIL images into NumPy arrays
# numpydata = asarray(img)
 
# # <class 'numpy.ndarray'>
# print(type(numpydata))
 
# #  shape
# print(numpydata.shape)

# nnumpydata = np.resize(numpydata, (224,224))
# print(nnumpydata.shape)
# model.predict(nnumpydata)


import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
model = keras.models.load_model("my_model.h5")

imgPath='DATASET\TRAIN\O\O_12118.jpg'

from PIL import Image
from numpy import asarray
 
 
# load the image and convert into
# numpy array
img = Image.open(imgPath)
test_datagen  = ImageDataGenerator(rescale = 1.0 / 255.0)

numpydata = asarray(img)
nnumpydata = np.resize(numpydata, (224,224,3))
nnumpydata = np.resize(numpydata, [1,224,224,3])
test_data = test_datagen.flow(nnumpydata,batch_size = 1)

# asarray() class is used to convert
# PIL images into NumPy arrays

 
# <class 'numpy.ndarray'>
print(type(numpydata))
 
#  shape
print(numpydata.shape)

nnumpydata = np.resize(numpydata, (224,224,3))
nnumpydata = np.resize(numpydata, [1,224,224,3])
print(nnumpydata.shape)
# k=(model.predict(nnumpydata))
k=(model.predict(test_data))
print(k)
# print(np.round_(k))
# k=k[0][0]
# print(k)
# s = "Object is recyclable" if k < 0.1 else "Object is not recyclable"
# print(k)
print("done")
# model.predict(nnumpydata)