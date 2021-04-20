# -*- coding: utf-8 -*-
"""beans2_1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1EpZn-LMJ8dtIM464s15NXpMgkgsYYopi
"""

import matplotlib.pylab as plt
import cv2
import tensorflow as tf
import tensorflow_hub as hub
import os
import numpy as np
import tensorflow_datasets as tfds
import warnings
from keras.optimizers import SGD
warnings.filterwarnings('ignore')

(x_train, y_train), (x_test, y_test),(x_val,y_val) = tfds.load(
    "beans",
    split=['train', 'test','validation'],
    batch_size=-1,
    as_supervised=True,
)

# # Resizing
x_train = tf.image.resize(x_train, [350,350]) 
x_val = tf.image.resize(x_val,[350,350])
x_test = tf.image.resize(x_test,[350,350])

print(x_train.shape)
print(x_val.shape)
print(x_test.shape)

# # Normalise the data
x_train = x_train/255
x_test = x_test/255
x_val = x_val/255

#implementing one hot encoding
from keras.utils.np_utils import to_categorical
ytrain = to_categorical(y_train, num_classes=3)
yval = to_categorical(y_val, num_classes=3)
ytest = to_categorical(y_test, num_classes=3)

#importing the model
from keras.models import Sequential
#creating model object
model=Sequential()

#importing layers
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout

#adding layers and forming the model
model.add(Conv2D(32,kernel_size=5,strides=1,padding="Same",activation="relu",input_shape=(350,350,3)))
model.add(Conv2D(32,kernel_size=5,strides=1,padding="Same",activation="relu"))
model.add(Conv2D(32,kernel_size=5,strides=1,padding="Same",activation="relu"))
model.add(MaxPooling2D(padding="same"))

model.add(Conv2D(32,kernel_size=5,strides=1,padding="Same",activation="relu"))
model.add(Conv2D(32,kernel_size=5,strides=1,padding="Same",activation="relu"))
model.add(Conv2D(32,kernel_size=5,strides=1,padding="Same",activation="relu"))
model.add(MaxPooling2D(padding="same"))

model.add(Conv2D(32,kernel_size=5,strides=1,padding="Same",activation="relu"))
model.add(Conv2D(32,kernel_size=5,strides=1,padding="Same",activation="relu"))
model.add(Conv2D(32,kernel_size=5,strides=1,padding="Same",activation="relu"))
model.add(MaxPooling2D(padding="same"))

model.add(Conv2D(32,kernel_size=5,strides=1,padding="Same",activation="relu"))
model.add(Conv2D(32,kernel_size=5,strides=1,padding="Same",activation="relu"))
model.add(Conv2D(32,kernel_size=5,strides=1,padding="Same",activation="relu"))
model.add(MaxPooling2D(padding="same"))

model.add(Conv2D(32,kernel_size=7,strides=1,padding="Same",activation="relu"))
model.add(Conv2D(32,kernel_size=7,strides=1,padding="Same",activation="relu"))
model.add(Conv2D(32,kernel_size=7,strides=1,padding="Same",activation="relu"))
model.add(MaxPooling2D(padding="same"))


model.add(Flatten())

model.add(Dropout(0.31))

model.add(Dense(3,activation="softmax"))
model.compile(optimizer="adamax" ,loss="categorical_crossentropy",metrics=["accuracy"])

model.summary()

history = model.fit(x_train,ytrain,batch_size=32,epochs=30,validation_data=(x_val,yval))

#model train and test scores
model.evaluate(x_val,yval),model.evaluate(x_train,ytrain)
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.legend();

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.legend();

test_loss, test_accuracy = model.evaluate(x_test,ytest)
print('Final testing accuracy ', test_accuracy)