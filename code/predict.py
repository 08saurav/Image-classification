import matplotlib.pyplot as plt
import skimage as s
from keras.preprocessing.image import load_img, array_to_img, img_to_array
import os
import numpy as np
import seaborn as sns
import pandas as pd
import keras
import argparse
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
import tensorflow as tf
import math
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required = True)
args = vars(parser.parse_args())

model = Sequential()
#filters,kernel_size,strides=(1, 1),padding='valid',data_format=None,dilation_rate=(1, 1),activation=None,use_bias=True,
#kernel_initializer='glorot_uniform',bias_initializer='zeros',kernel_regularizer=None,bias_regularizer=None,
#activity_regularizer=None,kernel_constraint=None,bias_constraint=None,

#pool_size=(2, 2), strides=None, padding='valid',data_format=None

model.add(Conv2D(32, (3,3),padding='same',activation='relu',input_shape=input_shape,name='conv2d_1'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),name='maxpool2d_1'))
model.add(Conv2D(32, (3,3),padding='same',activation='relu',name='conv2d_2'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Conv2D(64, (3, 3),activation='relu',name='conv2d_3'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),name='maxpool2d_2'))

model.add(Dropout(0.5))

model.add(Conv2D(128, (3, 3),padding='same',activation='relu',name='conv2d_5'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3),padding='same',activation='relu',name='conv2d_6'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(128, (3, 3),padding='same',activation='relu',name='conv2d_7'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))


model.add(Flatten())
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(3,activation='sigmoid'))

model.summary()

# load weights into new model
model.load_weights("../artifacts/first_try.h5")
def load_img_shapes(path_to_img):
    return cv2.imread(path_to_img).shape

def load_img(path_to_img):
    return cv2.imread(path_to_img)


img_path = args["path"]
img = image.load_img(img_path, target_size=(192, 192))
img = image.img_to_array(img)[:,:,0]                    # (height, width, channels)
img = np.expand_dims(img, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
img /= 255.


predictions = model.predict_classes(img)

print(predictions)
