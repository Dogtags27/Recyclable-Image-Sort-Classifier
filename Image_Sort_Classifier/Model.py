# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow

# import skimage.io
# import tqdm
# import glob

# from tqdm import tqdm
# from skimage.io import imread, imshow
# from skimage.transform import resize

# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications.vgg16 import VGG16
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import InputLayer, Dense, Flatten, Dropout, Activation, BatchNormalization
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from tensorflow.keras.preprocessing.image import load_img, img_to_array

# train_o = glob.glob('./DATASET/TRAIN/O/*.jpg')
# a=len(train_o)

# train_r = glob.glob('./DATASET/TRAIN/R/*.jpg')
# b=len(train_r)

# train_datagen = ImageDataGenerator(rescale= 1.0 / 255.0, zoom_range= 0.4, rotation_range= 10, horizontal_flip=True, vertical_flip=True, validation_split=0.2)

# valid_datagen = ImageDataGenerator(rescale= 1.0 / 255.0, validation_split=0.2)

# test_datagen = ImageDataGenerator(rescale= 1.0 / 255.0)


# train_dataset = train_datagen.flow_from_directory(directory='./DATASET/TRAIN/', target_size=(224, 224), class_mode= 'binary', batch_size= 128, subset= 'training')

# valid_dataset = valid_datagen.flow_from_directory(directory='./DATASET/TRAIN/', target_size=(224,224), class_mode= 'binary', batch_size= 128, subset= 'validation')


# fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(20,20))

# for i in tqdm(range(5)):
#     rand1 = np.random.randint(len(train_dataset))
#     rand2 = np.random.randint(128)
#     ax[i].imshow(train_dataset[rand1][0][rand2])
#     ax[i].axis('off')
#     label = train_dataset[rand1][1][rand2]
#     if label == 1:
#         ax[i].set_title('Recycle Waste')
#     else:
#         ax[i].set_title('Organic Waste')
        
# base_model = VGG16(input_shape=(224,224,3), include_top=False, weights="imagenet")
# for layer in base_model.layers:
#     layer.trainable = False
    
# model = Sequential()
# model.add(base_model)
# model.add(Dropout(0.2))
# model.add(Flatten())
# model.add(BatchNormalization())
# model.add(Dense(1024,kernel_initializer='he_uniform'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(Dense(1024,kernel_initializer='he_uniform'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(Dense(1,activation='sigmoid'))

# OPT = tensorflow.keras.optimizers.Adam(learning_rate=0.001)

# model.compile(loss='binary_crossentropy', metrics=[tensorflow.keras.metrics.AUC(name= 'auc')], optimizer=OPT)

# model.fit(train_dataset, validation_data=valid_dataset, epochs=15, verbose=1)

# import pickle

# Image_sort = pickle.dump(model)

